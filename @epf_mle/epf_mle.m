classdef epf_mle
    
    properties (SetAccess=protected)
        xGrid
        yGrid
        epf_fn
        xHat
        yHat
        entropy
        pullers
    end
    
    properties (SetAccess=public)
        xSpace
        ySpace
        stats
        computeEntropy = false;
        experts@expert
    end
    
    properties (Dependent=true)
        nDims
        nConditions
        nTime
        nExperts
        time
        testTime %Subset of time vector, containing only values in the test time points.
    end
    
    methods (Access=public)
        
        function d=epf_mle(xSpace,ySpace,varargin)
            
            %The epf_mle class is used to perform maximum-likeihood decoding of eye-position from population responses (i.e.
            %spike counts across N neurons). The decoding assumes Poisson noise (e.g., see addExpert() )
            %
            %Inputs:
            % xSpace and ySpace = the operational domain of decoding
            
            pin=inputParser;
            pin.addRequired('xSpace');
            pin.addRequired('ySpace');
            pin.addParameter('entropy',false);
            pin.parse(xSpace,ySpace,varargin{:});
            d.xSpace = xSpace;
            d.ySpace = ySpace;
            [d.xGrid, d.yGrid] = meshgrid(xSpace,ySpace);
            d.computeEntropy = pin.Results.entropy;
        end
        
        function d=addExpert(d,ex,ey,spkCounts,varargin)
            
            %Add a neuron to the population
            %
            % INPUTS:
            %
            % ex, ey        =   nConditions x nTimepoints matrices of eye-positions
            % spkCounts     =   1 x nConditions cell array. Each cell contains a nTrials x nTimepoints matrix of spike counts.
            %
            %                   OR
            %
            %                   nTrials x nTimepoints matrix. In this format, you MUST also specify a nTrials-length vector of
            %                   condition numbers using the 'conditions' param/value pair
            
            pin=inputParser;
            pin.addRequired('d');
            pin.addRequired('ex',@isnumeric);
            pin.addRequired('ey',@isnumeric);
            pin.addRequired('spkCounts',@(x) iscell(x) | isnumeric(x));
            pin.addParameter('conditions',@isnumeric);
            pin.addParameter('time',@isnumeric);
            pin.KeepUnmatched = true;
            pin.parse(d,ex,ey,spkCounts);
            
            %Create the expert object
            e = expert();
            
            %If entropy measures requested, use normal probabilities in calculations rather than log(p).
            if d.computeEntropy
                varargin = horzcat(varargin,'probAsLog',false);
            end
            
            %Assign the spike counts for this expert.
            e = addSpkCounts(e,ex,ey,spkCounts,varargin{:});
            
            %Add the expert to the panel
            d.experts(d.nExperts+1) = e;
            
            %Check that the time vector matches any existing experts
            if d.time~=e.time
                error('Time mismatch between experts in the epf_mle object.')
            end
        end
        
        function [d,victims] = rmNonSig(d,alpha)
            %Removes experts that do not have a significant EPF (according to GLM)
            %alpha = alpha level for hypothesis test
            if ~exist('alpha','var')
                alpha = 0.05;
            end
            stay = arrayfun(@(e) e.epf.isSig(alpha),d.experts);
            d.experts(~stay)=[];
            victims =find(~stay);
        end
        
        function d = train(d,varargin)
            %Fit the pEPF to each neuron using the training trials
            %To save decoding time, p(eye | count) is computed and stored
            pin=inputParser;
            pin.KeepUnmatched = true;
            pin.parse(varargin{:}); 
            for i=1:d.nExperts
                d.experts(i) = train(d.experts(i),d.xGrid,d.yGrid,pin.Unmatched);
            end
        end
        
        function d = mle(d,varargin)
            
            % Decoding is performed repeatedly for N sets of M synthetic trials (i.e. cross-neuron correlations destroyed)
            % For each set, M trials are drawn at random from each neuron within a given condition.
            % Only trials labelled as belonging to the TEST set are elligible for selection [all trials by default]
            % If cross validation is used, a new train/test set is generated for each set and the experts are re-trained
            %
            %   If using cross-validation:
            %
            %   - Set nSets to > 1 and specify additional CV parameters (propInTrainSet,minNumTestTrials)
            %   - For leave-one-out CV, set propInTrainSet to 1, minNumTestTrials to 1, and nTrialsPerSet to 1
            %
            %   If not using cross-validation:
            %
            %   - Set nSets = 1 and nTrialsPerset to > 1
            
            %Perform MLE decoding across many cross-validation sets.
            pin=inputParser;
            pin.KeepUnmatched = true;
            pin.addRequired('d');
            pin.addParameter('useCV',false);
            pin.addParameter('propInTrainSet',1);  %Defaults implement leave-one-out cross validation
            pin.addParameter('minNumTestTrials',1);
            pin.addParameter('nTrialsPerSet',1);
            pin.addParameter('nSets',100);
            pin.addParameter('excludePullers',false);
            pin.addParameter('nWorkers',8);    %Parallel computing used. Specify the number of virtual cores.
            pin.parse(d,varargin{:});
            p = pin.Results;
            p = rmfield(p,'d');
            p.nConditions = d.experts(1).nConditions;
            p.nTime = numel(d.testTime);
            p.nWorkers = pin.Results.nWorkers;
            
            %TODO: Check that e.trainSet.time and e.testSet.timeflags are constant across experts.
            
            
            %Check entropy values to identify experts that have undue
            %influence on decoder estimates.
            if p.excludePullers
                d = excludePullers(d,p,pin.Unmatched);
            end
            
            if p.useCV
                % ==== CROSS-VALIDATION ====
                [decX, decY, entrpy, pulls] = mle_cv(d,p);
            else
                % ==== REGULAR DECODING ==== (i.e. no iterative training)
                if p.nSets > 1
                    warning('No need for nSets > 1 when cross-validation is not being used.');
                end
                p.nSets = 1;
                [d,decX, decY, entrpy, pulls] = mle_nocv(d,p);
            end
            
            %Reorganise the results into a more convenient format
            d.xHat = permute(decX,[2 4 1 3]);
            d.yHat = permute(decY,[2 4 1 3]);
            if d.computeEntropy
                d.entropy = permute(entrpy,[2 4 1 3]);
                d.pullers = permute(pulls,[2 5 3 1 4]);
            else
                %Dump the memory
                d.entropy = [];
                d.pullers = [];
            end
        end
        
        function d = newTrainTestSets(d,varargin)
            
            %Generate a new list of TRAIN and TEST trials for each expert
            for i=1:d.nExperts
                d.experts(i) = newTrainTestSet(d.experts(i),varargin{:});
            end
        end
        
        function popResp = populationResponse(d,nTrialsPerSet)
            %Returns a [nConditions x nTrialsPerSet x nTime x nExperts] matrix of spike counts
            popResp = zeros(d.nConditions,nTrialsPerSet,numel(d.testTime),d.nExperts);
            for i=1:d.nExperts
                popResp(:,:,:,i) = testResponse(d.experts(i),'nSamples',nTrialsPerSet);
            end
        end
        
        function [decX,decY,entropy,pullers] = decodeAllConditions(d,popResp)           
            
            [nConditions,nTrialsPerSet,nTime,nExperts]=size(popResp);
            
            %Move the first dimension to the end (doing this so I can pull out data for each condition without using squeeze)
            popResp = permute(popResp,[2 3 4 1]);
            
            %Decode each condition in turn
            [decX,decY,entropy] = deal(zeros(nConditions,nTrialsPerSet,nTime));
            pullers = zeros(nConditions,5,nTrialsPerSet,nTime);
            for i=1:d.nConditions
                [decX(i,:,:),decY(i,:,:),entropy(i,:,:),pullers(i,:,:,:)]=decode(d,popResp(:,:,:,i));
            end          
        end
        
        function [xHat,yHat,entropy,pullers] = decode(d,popResp)
            
            %The second argument is the counts to be decoded. If a matrix, decoded each row separately
            [nTrialsPerSet,nTime,nExperts] = size(popResp);
            
            %Retrieve the PDF over labels for each expert for each entry in counts
            if d.computeEntropy
                %Use slow method of taking products across pdfs
                popPDF = ones(numel(d.xSpace)*numel(d.ySpace),nTrialsPerSet,nTime,'single');
                entrop = ones(nExperts,nTrialsPerSet,nTime,'single');
                for i=1:nExperts
                    
                    %Probabilities are stored as prob and not log(prob)
                    pr = d.experts(i).probEyeGivenCount(popResp(:,:,i));
                    
                    %Entropy
                    entrop(i,:,:) = -sum(pr.*log(pr),1);
                    
                    %Perform the Product of Experts
                    popPDF = popPDF .* pr;
                    
                    %Normalise PDFs
                    popPDF = popPDF./repmat(sum(popPDF,1),size(popPDF,1),1);
                end
                
                [~,sortInds]=sort(entrop,1);
                pullers = sortInds(1:5,:);
                
                %Calculate entropy (need to replace zeros with aribtary tiny number)
                popPDF(popPDF==0) = 0.000001;
                entropy = -sum(popPDF.*log(popPDF),1);
            else
                %Use fast method of summing log(p)
                popPDF = zeros(numel(d.xSpace)*numel(d.ySpace),nTrialsPerSet,nTime,'single');
                for i=1:nExperts
                    %Perform the Product of Experts (sum of logLH). Probabilities are stored as log(prob)
                    popPDF = popPDF + d.experts(i).probEyeGivenCount(popResp(:,:,i));
                end
           
                entropy = nans(1,nTrialsPerSet,nTime);
                pullers = nans(5,nTrialsPerSet,nTime);
            end
            %Choose the label with the highest probability
            [~,MAPind] = max(popPDF,[],1);
            MAPind = squeeze(MAPind);
            [MAPsubY,MAPsubX]=ind2sub(size(d.xGrid),MAPind);
            xHat = d.xSpace(MAPsubX);
            yHat = d.ySpace(MAPsubY);
        end
          
        function d = replaceSpkCounts(d,expertInd,ex,ey,spkCounts,varargin)
            pin=inputParser;
            pin.addRequired('expertInd',@isnumeric);
            pin.addRequired('ex');
            pin.addRequired('ey');
            pin.addRequired('spkCounts');
            pin.parse(expertInd,ex,ey,spkCounts);
            
            d.experts(expertInd) = addSpkCounts(d.experts(expertInd),ex,ey,spkCounts,varargin{:});
        end
        
        function d = summStats(d,varargin)
            %Compute summary statistics of decoder performance.
            
            pin=inputParser;
            pin.addRequired('d');
            pin.addParameter('from',d.time(1));
            pin.addParameter('to',d.time(end));
            pin.addParameter('analysisLabel','a');
            pin.addParameter('poolOverTime',false);
            pin.addParameter('reassignedConds',1:d.nConditions);  %Useful for pooling decoder estimates prior to performing analyses (d.stats) such as "quiver"
            %newConditions is a vector of length = nCurrentConditions containing new condition numbers.
            %Conditions assigned the same new number will be pooled, reducing the total number of conditions.
            pin.parse(d,varargin{:});
            from = pin.Results.from;
            to = pin.Results.to;
            poolOverTime = pin.Results.poolOverTime;
            analysisLabel = pin.Results.analysisLabel;
            reassignedConds = pin.Results.reassignedConds;
            if numel(reassignedConds)~=d.nConditions
                error('New condition specification must be a vector of length equal to nConditions');
            end
            
            %Pull out the data from the apprioriate timepoints
            timeInds = d.testTime >= from  & d.testTime <=to;
            timeVals = d.testTime(timeInds);
            decX = d.xHat(:,timeInds,:);
            decY = d.yHat(:,timeInds,:);
            if ~isempty(d.entropy)
                entropy = d.entropy(:,timeInds,:);
                hasEntropy = true;
            else
                entropy = [];
                hasEntropy = false;
            end
            ex = d.experts(1).ex(:,timeInds);
            ey = d.experts(1).ey(:,timeInds);
            
            %Renumber/pool conditions
            newConds = unique(reassignedConds);
            nConds = numel(newConds);
            for i=1:nConds;
                decXpool{i} = reshape(permute(decX(reassignedConds==newConds(i),:,:),[2 3 1]),size(decX,2),[]);
                decYpool{i} = reshape(permute(decY(reassignedConds==newConds(i),:,:),[2 3 1]),size(decX,2),[]);
                exPool(i,:) = mean(ex(reassignedConds==newConds(i),:),1);
                eyPool(i,:) = mean(ey(reassignedConds==newConds(i),:),1);
                if hasEntropy
                    entropyPool{i} = reshape(permute(entropy(reassignedConds==newConds(i),:,:),[2 3 1]),size(entropy,2),[]);
                else
                    entropyPool = [];
                end
            end
            
            %Pool over CV sets, trials within a CV set, and time (if requested)
            if poolOverTime
                decXpool = cellfun(@(x) reshape(x,1,[]),decXpool,'uniformoutput',false);
                decYpool = cellfun(@(y) reshape(y,1,[]),decYpool,'uniformoutput',false);
                exPool = mean(exPool,2);
                eyPool = mean(eyPool,2);
                timeVals = mean(timeVals);
                if hasEntropy
                    entropyPool = cellfun(@(x) reshape(x,1,[]),entropyPool,'uniformoutput',false);
                end
            end
            
            %Calculate statistics over the last dimension of the data matrices
            d.stats.(analysisLabel) = statistics(timeVals,decXpool,decYpool,exPool',eyPool',entropyPool,d.xSpace,d.ySpace);  
        end
        
        function d = excludePullers(d,p,varargin)
            pulls = findPullers(d,p,varargin{:});
            d.experts(pulls)=[];
            disp(['The following experts are pullers and have been removed: ', num2str(pulls)]);
        end
        
        function [pullers,pullStrength,entr] = findPullers(d,p,varargin)
            %Returns the entropy for every spike count in the test trials for every neuron.
            %Intended to be used as a way to identify neurons that have undue influence on the decoder.
            
            pin = inputParser;
            pin.addRequired('d');
            pin.addRequired('p');
            pin.addParameter('pullerPerc',10);     %The top X% are defined as pullers.
            pin.addParameter('plot',false);
            pin.parse(d,p,varargin{:});
            p=pin.Results.p;
            
            entr = cell(1,d.nExperts);
            if p.useCV
                for i=1:p.nSets
                    
                    %Reassign train/test trials
                    d = newTrainTestSets(d,'propInTrainSet',p.propInTrainSet,'minNumTestTrials',p.minNumTestTrials);
                    
                    %Re-fit gain-fields based on new training set
                    d = train(d);
                    
                    %Get the entropy for all test trials for every expert
                    tmpEntr = entropyPerTrial(d);
                    
                    %Add it to the heap for each neuron, collapsing (hence "reshape") over time and trials
                    entr = cellfun(@(a,b) vertcat(a,reshape(cell2mat(b),[],1)),entr,tmpEntr,'uniformoutput',false);
                end
            else
                %Get the entropy for all test trials for every expert
                for j=1:d.nExperts
                    entr{j} = reshape(cell2mat(entropyPerTrial(d.experts(j))),[],1);
                end
            end
            
            %Each neuron's "pull" strength is captured by the entropy at
            %its 25th percentile. i.e., how strongly it pulls 1/4 of the time.
            pullStrength = cellfun(@(thisCell) prctile(thisCell,25,1),entr);
            cutOff = prctile(pullStrength,pin.Results.pullerPerc);
            pullers = find(pullStrength < cutOff);
            
            if pin.Results.plot
                %Create a histogram showing these pull strenghts over all cells
                histogram(pullStrength,round(0.1*d.nExperts));
                line([cutOff,cutOff],ylim,'color','r');
            end
        end
        
        function entr = entropyPerTrial(d)
            for j=1:d.nExperts;
                entr{j} = entropyPerTrial(d.experts(j));
            end
        end
        %% ================= PLOTTING FUNCTIONS ===================
        function [fig_h,h] = heatMap(d,varargin)
            %Plot a heatmap showing the distribution of decoded positions for each condition.
            pin=inputParser;
            pin.addParameter('from',d.time(1));
            pin.addParameter('to',d.time(end));
            pin.addParameter('conditions',1:d.nConditions);
            pin.addParameter('xBins',[]);
            pin.addParameter('yBins',[]);
            pin.addParameter('poolOverTime',true);
            pin.addParameter('xlim',[]);
            pin.addParameter('ylim',[]);
            pin.addParameter('fig_h',[]);
            pin.parse(varargin{:});
            p = pin.Results;
            
            %Create/switch figure
            if isempty(p.fig_h)
                p.fig_h = figure;
            end
            figure(p.fig_h);
            
            %Select the data from the requested timepoints
            theseTimes = d.testTime >= p.from & d.testTime <= p.to;
            timeVals = d.testTime(theseTimes);
            decX = d.xHat(p.conditions,theseTimes,:);
            decY = d.yHat(p.conditions,theseTimes,:);
            
            if numel(unique(diff(timeVals)))>1
                error('Time bins must be contiguous.');
            end
            
            if p.poolOverTime && isempty(p.xlim)
               p.xlim = [min(d.xSpace),max(d.xSpace)];
            end
            
            if isempty(p.ylim)
                p.ylim = [min(d.ySpace),max(d.ySpace)];
            end
            
            if ~p.poolOverTime && isempty(p.xlim)
                p.xlim = [p.from,p.to];
            end
            
            %Get the eye position for each condition
            ex = d.experts(1).ex(p.conditions,theseTimes);
            ey = d.experts(1).ey(p.conditions,theseTimes);
            
            %Create subplot for each conditions, spatially arranged to match eye positions
            h = subplots(d,'conditions',p.conditions);
            
            %Create heatmap, over 2D space, or space and time.
            if p.poolOverTime
                %Pool over all timepoints
                decX = decX(:,:);
                decY = decY(:,:);
                
                %Average the eye positions over time (will have no effect if eye constant)
                ex = mean(ex,2);
                ey = mean(ey,2);
                
                %Set up histogram bins
                if isempty(p.xBins), p.xBins = d.xSpace; end
                if isempty(p.yBins), p.yBins = d.ySpace; end
                
                %Create the heat-map for each condition
                for i=1:numel(p.conditions)
                    
                    %Switch focus to current plot
                    axes(h(i));
                    
                    %Calculate 2D histogram for the decoder samples
                    [n,bins] = hist3([decX(i,:)',decY(i,:)'],{p.xBins,p.yBins});
                    
                    %Show as image
                    imagesc(bins{1},bins{2},n');
                    colormap('hot'); set(gca,'ydir','normal'); hold on;  axis equal;
                    xlim(p.xlim); ylim(p.ylim);
                    
                    %Apply a grey grid to plots.
                    set(gca,'xgrid','on','xcolor',[0.5,0.5,0.5],'ygrid','on','ycolor',[0.5,0.5,0.5]);
                    plot(ex,ey, 'o','markersize', 8, 'markerfacecolor', [0 0.5 1],'markeredgecolor', [0 0.15 0.3],'linewidth',1.5); hold on
                    plot(ex(i),ey(i), 'o','markersize', 8, 'markerfacecolor', [0 0.8 0],'markeredgecolor', [0 0.3 0],'linewidth',1.5);
                end
            else
                %Set up histogram time bins
                if isempty(p.xBins), p.xBins = timeVals; end
                if isempty(p.yBins), p.yBins = d.ySpace; end
                set(gcf,'Units','Normalized');
                plotWidth =0.2;
                %Create the heat-map for each condition
                for i=1:numel(p.conditions)
                    
                    %Get the position of the current plot.
                    plotPos = get(h(i),'position');

                    %Replace it with a split plot
                    delete(h(i));
                    
                    %X-coord heatmap
                    axes('Position',[plotPos(1),plotPos(2)+plotWidth/2,plotWidth,plotWidth/2]);
                    x_h(i) = spaceTimeHist(d,timeVals,squeeze(decX(i,:,:)),p.xBins,p.yBins); set(x_h(i),'xticklabel','');
                    plot(timeVals,ex(i,:),'c','linewidth',2);
                    xlim(p.xlim); ylim(p.ylim);
                    
                    %Y-coord heatmap
                    axes('Position',[plotPos(1),plotPos(2),plotWidth,plotWidth/2]);
                    y_h(i) = spaceTimeHist(d,timeVals,squeeze(decY(i,:,:)),p.xBins,p.yBins);
                    plot(timeVals,ey(i,:),'c','linewidth',2);
                    xlim(p.xlim); ylim(p.ylim);
                end              
                colormap('hot');
            end
            fig_h = p.fig_h;
        end
        
        function fig_h = quiver(d,anal,varargin)
            %Quiver plot showing accuracy of decoded position for each eye
            %position. Currently uses medians only.
            pin=inputParser;
            pin.addRequired('d');
            pin.addRequired('anal');
            pin.addParameter('errorType', 'BARS', @(x) any(strcmpi({'BARS','ELLIPSE'},x)));
            pin.addParameter('fig_h', []);
            pin.addParameter('color',[0 0 0]);
            pin.addParameter('title', []);
            pin.parse(d,anal,varargin{:});
            p = pin.Results;
            
            fig_h = p.fig_h;
            if isempty(fig_h)
                fig_h = figure;
            end
            
            if ~strcmpi(class(fig_h),'matlab.ui.Figure')
                error('Figure handle argument is not a figure handle!');
            end
            
            if numel(d.stats.(anal).time)>1
                error('quiver currently only supports stats analyses with a single time-point');
            end       
            
            figure(fig_h);
            ex = d.stats.(anal).ex;
            ey = d.stats.(anal).ey;
            x = d.stats.(anal).X;
            y = d.stats.(anal).Y;
            
            plot(ex,ey,'k+', 'markersize', 16,'linewidth',2,'markeredgecolor',[0 0.8 0]); hold on
            switch upper(p.errorType)
                case 'BARS'
                    h=herrorbar(x.p50,y.p50,x.p50-x.p25,x.p75-x.p50,'ko'); set(h, 'markersize', 9, 'markeredgecolor',[0 0 0],'markerfacecolor',p.color); hold on;
                    h=errorbar(x.p50,y.p50,y.p50-y.p25,y.p75-y.p50,'ko'); set(h, 'markersize', 9, 'markeredgecolor',[0 0 0],'markerfacecolor',p.color);
                case 'ELLIPSE'
                    for i=1:numel(ex)
                        h = error_ellipse(squeeze(d.stats.(anal).covXY(:,i,:,:)),[x.p50(i),y.p50(i)]);
                        set(h,'color',[0.8 0.8 0.8]); hold on;
                        plot(x.p50,y.p50,'o','markersize', 9, 'markeredgecolor',[0 0 0],'markerfacecolor',p.color);
                    end
                otherwise
                    error('epf_mle:quiver - Unknown error bar type');
            end
            quiver(ex,ey,x.p50-ex,y.p50-ey,0,'.','linestyle',':','color',[0.5, 0.5,0.5]);
            xlabel('Eye position (X)'); ylabel('Eye position (Y)'); title(p.title); axis equal;
        end
        
        function plot_h = plotStat(d,analLabel,fieldName,varargin)
            pin=inputParser;
            pin.addRequired('analLabel');
            pin.addRequired('fieldName');
            pin.addParameter('plot_h',[]);
            pin.addParameter('centroidStat','mean',@(x) any(strcmpi(x,{'mean','median'})));
            pin.addParameter('conditions',[]);
            pin.addParameter('lineSpecs',{},@iscell);
            pin.addParameter('type','shaded',@(x) any(strcmpi(x,{'shaded','bars'})));
            pin.addParameter('perCondition',false) %Plot each condition or average over conditions.
            pin.parse(analLabel,fieldName,varargin{:});
            p=pin.Results;
            
            if isempty(p.plot_h)
                figure;
                p.plot_h = gca;
            end
            
            %Retrieve the analysis data
            data = d.stats.(analLabel);
            
            %Check that this analysis played out over time
            if numel(data.time)==1
                error('plotStats intended only for summStats() analyses that include multiple timepoints');
            end
            
            %By default, all conditions
            if isempty(p.conditions)
                p.conditions = 1:size(data.(fieldName).mean,2);
            end
            
            %Retrieve mdata to plot.
            switch upper(p.centroidStat)
                case 'MEAN'
                    stat = data.(fieldName).mean(:,p.conditions);
                    errorL = data.(fieldName).std(:,p.conditions);
                    errorU = data.(fieldName).std(:,p.conditions);
                case 'MEDIAN'
                    stat = data.(fieldName).p50(:,p.conditions);
                    errorL = data.(fieldName).p50(:,p.conditions)-data.(fieldName).p25(:,p.conditions);
                    errorU = data.(fieldName).p75(:,p.conditions)-data.(fieldName).p50(:,p.conditions);
                otherwise
                    error([p.centroidMeasure ' is an unknown measure of central tendency']);
            end
            
            %Average over conditions, if requested
            if ~p.perCondition
                errorL = std(stat,[],2)./sqrt(numel(p.conditions));
                errorU = std(stat,[],2)./sqrt(numel(p.conditions));
                stat = mean(stat,2);
            end
            
            %Plot the errorbars
            switch upper(p.type)
                case 'SHADED'
                    errors(:,1) = errorL;
                    errors(:,2) = errorU;
                    [hl, hp] = boundedline(d.time,stat,errors,'alpha',p.plot_h); hold on;
                    if ~isempty(p.lineSpecs)
                        set(hl,p.lineSpecs{:});
                    end
                    if any(strcmpi(p.lineSpecs,'color'))
                        col = p.lineSpecs{find(strcmpi(p.lineSpecs,'color'))+1};
                        set(hp,'faceColor',col);
                    end
                case 'BARS'
                    h = errorbar(p.plot_h,data.time,stat,errorL,errorU,'o','markersize',15,'markerfacecolor','w'); hold on;                    
                    %Apply line specs
                    if ~isempty(p.lineSpecs)
                        set(h,p.lineSpecs{:});
                    end
                otherwise
                    error('Unknown plot type');
            end
            
            xlim([min(data.time),max(data.time)]);
            xlabel('Time (ms)');
            plot_h = p.plot_h;
        end
        
        function h = subplots(d,varargin)
            %Generates subplots spatially arranged to match the true eye position for each condition.
            %h = vector of handles to each subplot
            pin=inputParser;
            pin.addParameter('conditions',1:d.nConditions);    %Defaults to initial eye position per condition
            pin.parse(varargin{:});
            p=pin.Results;
            
            %Get the eye positions for the specified conditions (using only the initial position)
            ex = d.experts(1).ex(p.conditions,1);
            ey = d.experts(1).ey(p.conditions,1);
            
            %Set up functions to position plots
            origUnits = get(gcf,'Units');
            set(gcf,'Units','Normalized');
            scaleFactor = max(max(ex)-min(ex),max(ey)-min(ey));
            plotX = (ex-min(ex))./scaleFactor;
            plotY = (ey-min(ey))./scaleFactor;
            pos_fn = @(x) 0.7*x +0.05;

            %Generate axes
            if numel(ex)>1
                for i=1:numel(ex)
                    h(i) = axes('position',[pos_fn(plotX(i)),pos_fn(plotY(i)),0.2,0.2]);
                end
            else
                h=axes;
            end
            set(gcf,'Units',origUnits);
        end
    end
    
    methods (Access = private)
        
        function [d,decX,decY,entropy,pullers] = mle_nocv(d,p)
            
            %Check whether the experts have already been been trained.
            isTrained = arrayfun(@(x) ~isempty(x.epf),d.experts);
            if any(~isTrained)
                warning('One or more experts in the epf_mle object is untrained. Training all (again?) now...');
                d = train(d);
            end
            
            %Pre-allocate storage for the decoded eye-positions
            [decX, decY, entropy] = deal(zeros(p.nSets,p.nConditions,p.nTrialsPerSet,p.nTime));
            pullers = zeros(p.nSets,p.nConditions,5,p.nTrialsPerSet,p.nTime);
            for i=1:p.nSets
                
                %Get population responses for N test trials, [nConditions x nTrialsPerSet x nTime x nExperts]
                popResp = populationResponse(d,p.nTrialsPerSet);
                
                %Decode the population responses
                [decX(i,:,:,:),decY(i,:,:,:),entropy(i,:,:,:),pullers(i,:,:,:,:)] = decodeAllConditions(d,popResp);
            end
        end
        
        function [decX,decY,entropy,pullers] = mle_cv(d,p)
            %Perform MLE repeatedly for N CV sets using parallel computing
            
            %Pre-allocate storage for the decoded eye-positions
            [decX, decY] = deal(zeros(p.nSets,p.nConditions,p.nTrialsPerSet,p.nTime));
            pullers = zeros(p.nSets,p.nConditions,5,p.nTrialsPerSet,p.nTime);
            
            %Need to allocate these to local variables to allow the parfor to work
            propInTrainSet = p.propInTrainSet;
            minNumTestTrials = p.minNumTestTrials;
            nTrialsPerSet = p.nTrialsPerSet;
            
            pool=parpool([1, p.nWorkers]);
            parfor i=1:p.nSets
                
                %Randomly sample a new TRAINING set and TEST set for each neuron
                d_tmp = newTrainTestSets(d,'propInTrainSet',propInTrainSet,'minNumTestTrials',minNumTestTrials);
                
                %Train the decoder using the training trials
                d_tmp = train(d_tmp);
                
                %Get population responses for N test trials, [nConditions x nTrialsPerSet x nTime x nExperts]
                popResp = populationResponse(d_tmp,nTrialsPerSet);
                
                %Decode the population responses
                [decX(i,:,:,:),decY(i,:,:,:),entropy(i,:,:,:),pullers(i,:,:,:,:)] = decodeAllConditions(d_tmp,popResp);
            end
            delete(pool);
        end
        
        function h = spaceTimeHist(d,time,decs,binX,binY)
            nSamples = size(decs,2);
            tMat = repmat(time(:),1,nSamples);
            [n,bins] = hist3([decs(:),tMat(:)],'Ctrs',{binY,binX});
            imagesc(bins{2},bins{1},n); hold all;
            plot(time,median(decs,2),'linewidth',3,'color',[1,1,1]);
            h=gca;
            set(h,'ydir','normal'); hold on;
        end
    end
    
    methods (Static)
        function plotXYvsTime(time,ex,ey,decX,decY,varargin)
            %Generic function to plot decoded X and Y data vs time, with
            %the real eye as a reference.
            %Doesn't actually use the epf_mle object at all.
            pin = inputParser;
            pin.addRequired('time');
            pin.addRequired('ex');
            pin.addRequired('ey');
            pin.addRequired('decX');
            pin.addRequired('decY');
            pin.addParameter('errorX',[]);
            pin.addParameter('errorY',[]);
            pin.addParameter('errorType','PATCH',@(x) any(strcmpi(x,{'PATCH','BARS'})));
            pin.addParameter('predTime',[]);    %Can include a *predicted* decoded eye position over time as a second reference.
            pin.addParameter('predX',[]);
            pin.addParameter('predY',[]);
            pin.addParameter('lineScalar',1);   %Scalar for the thickness of all lines.
            pin.addParameter('xlim',[min(time),max(time)]);
            pin.addParameter('ylim',[]);
            pin.addParameter('xColor',[]);
            pin.addParameter('yColor',[]);
            pin.parse(time,ex,ey,decX,decY,varargin{:});
            p = pin.Results;
            predTime = p.predTime;
            predX = p.predX;
            predY = p.predY;
            errorX = p.errorX;
            errorY = p.errorY;
            
            %Check format of inputs
            time=time(:); ex = ex(:); ey=ey(:); decX=decX(:); decY=decY(:);
            if ~isequal(size(errorX),size(errorY)) || (size(errorX,2)>2) || (size(errorY,2)>2)
                error('errorX and errorY must be the same size and eith m x 1 or m x 2');
            end
            
            %If using same value for upper and lower error values, duplicate into two columns
            if size(errorX,2)==1
                errorX = [errorX,errorX];
            end
            if size(errorY,2)==1
                errorY = [errorY,errorY];
            end
            
            cOrder = get(gca,'colororder');
            if isempty(p.xColor)
                p.xColor = cOrder(2,:);
            end
            if isempty(p.yColor)
                p.yColor = cOrder(1,:);
            end
            
            %Plot the ACTUAL eye position
            plot(time,ey,'linewidth',p.lineScalar*2,'color',[0.5 0.5 0.5]); hold on;
            plot(time,ex,'linewidth',p.lineScalar*2,'color',[0 0 0]);
            
            %Plot the PREDICTED position
            if ~isempty(predX)
                plot(predTime,predY,':','color',p.yColor,'linewidth',p.lineScalar*3);
                plot(predTime,predX,':','color',p.xColor,'linewidth',p.lineScalar*3);
            end
            
            %Plot the DECODED position           
            if isempty(errorX)
                plot(time,decY,'w','linewidth',p.lineScalar*4+1);
                plot(time,decY,'linewidth',p.lineScalar*4,'color',p.yColor);
                plot(time,decX,'w','linewidth',p.lineScalar*4+1);
                plot(time,decX,'linewidth',p.lineScalar*4,'color',p.xColor);
            else
                switch upper(p.errorType)
                    case 'BARS'
                        h = errorbar(time,decY,errorY(:,1),errorY(:,2),'o','color',p.yColor,'linewidth',1,'markersize',12,'markerfacecolor','w'); hold all;
                        h = errorbar(time,decX,errorX(:,1),errorX(:,2),'o','color',p.xColor,'linewidth',1,'markersize',12,'markerfacecolor','w');
                    case 'PATCH'
                        [hly,hpy] = boundedline(time,decY,errorY,'b','alpha'); set(hly,'linewidth',p.lineScalar*4);
                        [hlx,hpx] = boundedline(time,decX,errorX,'r','alpha'); set(hlx,'linewidth',p.lineScalar*4);
                        set(hlx,'color',p.xColor); set(hly,'color',p.yColor);
                        set(hpx,'facecolor',p.xColor); set(hpy,'facecolor',p.yColor);
                end
            end
            
            %Set axis limits
            xlim(p.xlim);
            if ~isempty(p.ylim)
                ylim(p.ylim);
            end
        end
    end
    methods
        function nConditions = get.nConditions(d)
            nConditions = size(d.experts(1).ex,1);
        end
        
        function nTime = get.nTime(d)
            nTime = numel(d.time);
        end
        
        function n = get.nExperts(d)
            n=numel(d.experts);
        end
        
        function val = get.time(d)
            e = d.experts(1);
            val = e.time;
        end
        
        function val = get.testTime(d)
            e = d.experts(1);
            val = e.time(e.testSet.times);
        end
    end
end