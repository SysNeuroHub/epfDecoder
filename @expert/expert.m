classdef expert
    
    properties
        comments;
        id;
        useGainCorr@logical=false;
    end
    
    properties (SetAccess=protected)
        time
        ex
        ey
        spkCounts
        epf@epfGLM
        trainSet
        testSet
        gainPerTime
        xGrid
        yGrid
    end
    
    properties (Dependent=true)
        nDims
        nConditions
    end
    
    properties (GetAccess=public, SetAccess=private)
        pdf;
        minSpikeCount;
        maxSpikeCount;
        pEyeGivenCount
        probAsLog
    end
    
    methods (Access=private)
        
    end
    
    methods
        
        function e=expert()
            % Stores neural data and pEPF data for a single neuron, used within the epf_mle class
        end
        
        function e = addSpkCounts(e,ex,ey,spkCounts,varargin)
            % INPUTS:
            %
            % ex, ey        =   nConditions x nTimepoints matrices of eye-positions
            % spkCounts     =   1 x nConditions cell array. Each cell contains a nTrials x nTimepoints matrix of spike counts.
            %
            %                   OR
            %
            %                   nTrials x nTimepoints matrix. In this format, you MUST also specify a nTrials-length vector of
            %                   condition numbers using the 'conditions' param/value pair
            % PARAM/VALUE PAIRS:
            %
            % isTrainTime   =   1 x nTimepoints logical vector flagging which timepoints should be used to fit the pEPF
            % isTestTime    =   1 x nTimepoints logical vector flagging which timepoints should be used during decoding
            % isTrainTrial  =   1 x nConditions cell array of logical vectors flagging which trials should be used for fitting
            % isTestTrial   =   1 x nConditions cell array of logical vectors flagging which trials should be used during decoding
            % conditions    =   1 x nTrials vector of condition numbers
            % probAsLog     =   true/false. Whether probabilities should be stored internally as p(x) or log(p(x)). Only used for code optimisation/speed.
            
            pin=inputParser;
            pin.addRequired('ex',@isnumeric);
            pin.addRequired('ey',@isnumeric);
            pin.addRequired('spkCounts',@(x) iscell(x) | isnumeric(x));
            pin.addParameter('time',[],@isnumeric);
            pin.addParameter('conditions',[],@isnumeric);
            pin.addParameter('probAsLog',true);
            pin.parse(ex,ey,spkCounts,varargin{:});
            conditions = pin.Results.conditions;
            timeVals = pin.Results.time;
            e.probAsLog = pin.Results.probAsLog;
            
            %Check data conforms to an accepted format
            if ~iscell(spkCounts)
                %Trial x time matrix format
                %Check that the number of trials in spkCounts and conditions matches
                if ~isempty(conditions)&&(numel(conditions)==size(spkCounts,1));
                    %Matches, so convert into cell format
                    %Convert condition numbers into 1:N (in case they weren't already)
                    [~,~,conditions] = unique(conditions);
                    for i=1:max(conditions)
                        cellFormat{i} = spkCounts(conditions==i,:); %#ok<AGROW>
                    end
                    spkCounts = cellFormat;
                else
                    error('spkCount and condition arguments do not match. Check input requirements for expert()');
                end
            end
            
            %Make sure that all trials have the same number of timepoints
            if ~(numel(unique(cellfun(@(x) size(x,2),spkCounts)))==1);
                error('The number of timepoints is inconsistent across trials/conditions');
            end
            
            %Check eye arguments
            if ~isequal(size(ex),[numel(spkCounts),size(spkCounts{1},2)])
                error('The eye position vector/matrix does not match neural data. Should be vertical vector or matrix');
            end
            
            if isempty(timeVals)
                timeVals = 0:size(spkCounts{1},2)-1;
            end
            
            %All checking done. Assign spike counts and eye data
            e.spkCounts = spkCounts;
            e.ex = ex;
            e.ey = ey;
            e.time = timeVals(:)';
            
            %Populate nTimes list of gain values with default values
            e = setGain(e);
            
            %Flag all trials for both training and testing
            e = flagTrials(e,'TRAIN');
            e = flagTrials(e,'TEST');
            
            %Flag all time points for both training and testing
            e = flagTimes(e,'TRAIN');
            e = flagTimes(e,'TEST');
            
            %Update the max and min spike count properties (not using dependent property for these because called zillions of times)
            e = setMaxAndMinSpikeCount(e);
            
            %If the PDFs/EPFs have already been computed/fitted, re-compute the PDFs to ensure the range of new spike counts is covered
            if ~isempty(e.pEyeGivenCount)
                e=computePDFs(e);
            end
        end
        
        function e = setGain(e,vals,varargin)           
            if isempty(vals)
                vals = ones(size(e.time));
            end
            
            if any(vals <= 0)
                error(['Gain values cannot be 0 or below 0, element(s) (',num2str(find(~vals)),') is/are ' num2str(vals<=0)])
            end
            
            if numel(vals) ~= numel(e.time)
                error('Gain must be a vector with a length equal to the number of time points')
            end
            
            e.gainPerTime = vals(:)';
            
            %If the expert has already been trained, we now need to re-compute the PDFs with the new gain values
            if ~isempty(e.epf)
                e = computePDFs(e);
            end
        end
        
        function e = flagTimes(e,setType,varargin)
            %Explain usage options. requirements.
            
            pin=inputParser;
            pin.addRequired('e');
            pin.addRequired('setType',@(x) any(strcmpi(x,{'TRAIN', 'TEST'})));
            pin.addParameter('from',[]);
            pin.addParameter('to',[]);
            pin.addParameter('flags',[],@(x) isequal(size(x),size(e.time)));
            %pin.addParameter('flags',[],@(x) islogical(x) && isequal(size(x),size(e.time)));
            pin.parse(e,setType,varargin{:})
            p = pin.Results;
            setType = lower(setType);
            setType = strcat(setType,'Set');
            
            %If a logical vector of times is provided, use instead of p.from and p.to values
            if ~isempty(p.flags)
                if ~isempty(p.from) || ~isempty(p.to)
                    error('dfhidfhosidf');
                end
                times = p.flags;
            else
                %If time windows are not provided set p.from to the first value in e.time and p.to to the last
                if isempty(p.from)
                    p.from = e.time(1);
                end
                if isempty(p.to)
                    p.to = e.time(end);
                end
                
                %Check that input training and testing windows match up with data times
                if numel(intersect([p.from,p.to],e.time))<2 && p.to ~= p.from
                    error([p.setType,' window must specify only those times for which there are data, ',num2str(setdiff([p.from,p.to],e.time)),' is/are not legal'])
                end
                
                %Produce logical vectors of time periods that are used for training and testing if those windows are provided
                if ~isempty([p.from,p.to])
                    times = e.time >= p.from & e.time <= p.to;
                end
            end
            
            %Add logicals to expert object
            e.(setType).times = times;
            
            %If the expert has already been trained, we now need to re-compute the PDFs with the new gain values
            if ~isempty(e.epf) && strcmpi(setType,'test')
                e = computePDFs(e);
            end
        end
        
        function e = flagTrials(e,setType,varargin)
            
            % Function that flags trials. Can be used to set either train trials or test trials, depending
            % whether the input argument is set to 'TRAIN' or 'TEST'
            
            %Trials are used for training (i.e. fitting the pEPF)/testing only if
            %their entry in e.(taskType).trials is set to true
            
            pin=inputParser;
            pin.addRequired('e');
            pin.addParameter('trialFlags',[],(@(x) isempty(x) || iscell(x)));
            pin.addRequired('setType',@(x) any(strcmpi(x,{'TRAIN', 'TEST'})));
            pin.parse(e,setType,varargin{:})
            setType = lower(setType);
            setType = strcat(setType,'Set');
            trialFlags = pin.Results.trialFlags;
            
            %Check whether any selections have been requested
            if isempty(trialFlags)
                % No particular trials specified. Use ALL trials for training by default
                e.(setType).trials = cellfun(@(x) true(size(x,1),1),e.spkCounts,'UniformOutput',false);
                return;
            end
            
            %If trial indices are specified rather than logical vectors
            if ~islogical(trialFlags)
                
                %Reset all trials to FALSE
                e.(setType).trials = cellfun(@(x) false(size(x,1),1),e.spkCounts,'UniformOutput',false);
                
                %Set the requested trials to TRUE
                for i=1:numel(e.(setType).trials)
                    e.(setType).trials{i}(trialFlags{i})=true;
                end
            else
                %Check that the logical input is the right size
                sizeFun = cellfun(@size,trialFlags,'UniformOutput',false);
                if ~isequal(cellfun(sizeFun,e.(setType).trials),cellfun(sizeFun,trialFlags))
                    error(['There is a mismatch between the spike-count data and the requested ' setType ' trials']);
                else
                    e.(setType).trials = trialFlags;
                end
            end
        end
        
        function e=train(e,xGrid,yGrid,varargin)
            
            %Compute each neuron's probabilistic eye-position field (pEPF).
            pin=inputParser;
            pin.KeepUnmatched = true;
            pin.addRequired('e');
            pin.addRequired('xGrid');
            pin.addRequired('yGrid');
            pin.parse(e,xGrid,yGrid,varargin{:});
            e.xGrid = xGrid;
            e.yGrid = yGrid;
            
            %Store the eye positions for training separately
            e.trainSet.ex = e.ex(:,e.trainSet.times);
            e.trainSet.ey = e.ey(:,e.trainSet.times);
            
            %Check that there is only one eye position for each condition in the training data
            eyePosIsUnique = all(all(diff(e.trainSet.ex,[],2)==0,2)) && all(all(diff(e.trainSet.ey,[],2)==0,2));
            if eyePosIsUnique
                e.trainSet.ex = e.trainSet.ex(:,1); %All timepoints have same eye-position. Just take the first.
                e.trainSet.ey = e.trainSet.ey(:,1);
            else
                error('*** Training data contains more than one eye position within a condition. Not permitted. ***');
            end
            
            %Retrieve the spiike counts on the training trials and training times
            counts = cellfun(@(cts,tr) reshape(cts(tr,e.trainSet.times),1,[]),e.spkCounts,e.trainSet.trials,'uniformoutput',false);
            
            %Do the fit
            e.epf = epfGLM(pin.Unmatched);
            e.epf = glm(e.epf,e.trainSet.ex,e.trainSet.ey,counts);
            
            %Pre-compute the likelihood and posterior probability functions
            e=computePDFs(e);
        end
        
        function e = computePDFs(e)
            
            %Clear pEyeGivenCount before repopulating
            e.pEyeGivenCount = [];
            
            %gain Inds should be set to 1 if there is only one gain value or if gain correction isn't being used
            if numel(e.gainPerTime)==1 || ~e.useGainCorr
                gainInds = 1;
            else
                gainInds = find(e.testSet.times);
            end
            
            for i = 1:numel(gainInds)
                
                %if not using gain correction PDFs should not use the gain value listed in e.gainPerTime.
                %Use proxy variable and replace with 1.
                if ~e.useGainCorr
                    gainVals = 1;
                else
                    gainVals = e.gainPerTime;
                end
                
                %Evaluate the epf at each position in the grid, multiplied by the gain of the neuron at that test time
                vals = gainVals(gainInds(i))*e.epf.feval(e.xGrid(:),e.yGrid(:));
                
                %What is the range of spike counts to use?
                theseCounts = e.minSpikeCount:e.maxSpikeCount;
                
                %Compute p(count | eye) for all points in eye space
                pCountGivenEye = poisspdf(repmat(theseCounts,numel(vals),1),repmat(vals,1,numel(theseCounts)));
                
                %Convert p(count|ex,ey) to p(ex,ey|count). i.e. normalise PDF over eye position space for each spike count (so it sums to 1)
                e.pEyeGivenCount(:,:,i) = pCountGivenEye./repmat(sum(pCountGivenEye,1),size(pCountGivenEye,1),1);
                
                %Store probabilities
                e.pEyeGivenCount(:,:,i) = single(e.pEyeGivenCount(:,:,i));
                
                %Store as log(pr).
                if e.probAsLog
                    e.pEyeGivenCount(:,:,i) = log(e.pEyeGivenCount(:,:,i));
                end
            end
        end
        
        function prob = probEyeGivenCount(e,count)
            
            countInds = c2i(e,count);
            nSpace = size(e.pEyeGivenCount,1);
            [nSamples,nTestTimes] = size(count);
            prob = zeros(nSpace,nSamples,nTestTimes);
            
            %If we aren't using time-dependent gain values, return eye PDFs for each sample x time
            if size(e.pEyeGivenCount,3) == 1 || ~e.useGainCorr
                prob = reshape(e.pEyeGivenCount(:,countInds(:),1),nSpace,nSamples,nTestTimes);
            else
                %Otherwise, use the appropriate page in the pdf for each time point
                for i=1:nTestTimes
                    prob(:,:,i) = e.pEyeGivenCount(:,countInds(:,i),i);
                end
            end
        end
        
        function e = newTrainTestSet(e,varargin)
            
            %Re-assign flags for training and test sets based on cross-validation type.
            pin=inputParser;
            pin.addRequired('e');
            pin.addParameter('propInTrainSet',1,@(x)(x>=0)&&(x<=1));    %Default settings here are leave-one-out cross validation
            pin.addParameter('minNumTestTrials',1);
            pin.parse(e,varargin{:})
            propInTrainSet = pin.Results.propInTrainSet;
            minNumTestTrials = pin.Results.minNumTestTrials;
            
            %Randomly select trials for each condition
            for i=1:e.nConditions
                %How many trials in total for the current condition?
                nTrials = size(e.spkCounts{i},1);
                
                %How many train and test trials should be used?
                nTrain = min(ceil(propInTrainSet*nTrials),nTrials-minNumTestTrials);
                nTest = max(nTrials-nTrain,minNumTestTrials);
                
                %Set the appropriate flags
                if (nTrain + nTest) == nTrials
                    %Generate the train flags and shuffle them randomly
                    theseTrainFlags = [true(nTrain,1); false(nTest,1)];
                    trainFlags{i} = randsample(theseTrainFlags,nTrials);
                    testFlags{i} = ~trainFlags{i};
                else
                    error('There are insufficient trials to perform cross-validation');
                end
            end
            
            %Assign the flags to the expert
            e = flagTrials(e,'TRAIN','trialFlags',trainFlags);
            e = flagTrials(e,'TEST','trialFlags',testFlags);
        end
        
        function response = testResponse(e,varargin)
            pin=inputParser;
            pin.addRequired('e');
            pin.addParameter('eyeConds',1:e.nConditions);
            pin.addParameter('nSamples',1);
            pin.parse(e,varargin{:})
            eyeConds = pin.Results.eyeConds;
            nSamples = pin.Results.nSamples;
            
            nConds = numel(eyeConds);
            nTime = sum(e.testSet.times);
            response = zeros(nConds,nSamples,nTime);
            for i=1:numel(eyeConds)
                %Return the samples (over time) from all the test trials for the current condition
                theseSamples = e.spkCounts{i}(e.testSet.trials{i},e.testSet.times);
                
                nTestTrials = size(theseSamples,1);
                
                %Randomly select nSamples trials (rows) with replacement
                response(i,:,:) = theseSamples(randsample(1:nTestTrials,nSamples,true),:);
            end
            
        end
        
        function e = setMaxAndMinSpikeCount(e)
            %Find the maximum spike count in the entire set
            globalMin = @(x) min(x(:));
            e.minSpikeCount = min(cellfun(globalMin,e.spkCounts));
            globalMax = @(x) max(x(:));
            e.maxSpikeCount  = max(cellfun(globalMax,e.spkCounts));
        end
        
        function [out,fig_h] = fitEPF(e,varargin)
            %Fit an epf (regression) to the spike counts for a given time interval
            %Note, this function is not used as part of decoding. It's for
            %analysis/plotting of gain fields.
            %
            %The 'reassignedConds' param/value pair is useful for re-ordering/pooling conditions.
            %Pass a vector of length equal to the number of current conditions in which each entry
            %specifies the new condition number.  %Conditions assigned the same new number will be pooled, reducing the total number of conditions
            pin=inputParser;
            pin.KeepUnmatched = true;
            pin.addParameter('from',[]);
            pin.addParameter('to',[]);
            pin.addParameter('timeFlags',[]); %Logical vector to choose particular time-points
            pin.addParameter('reassignedConds',1:e.nConditions);
            pin.addParameter('fig_h',[]);
            pin.addParameter('plot',true);
            pin.addParameter('nPlotHoops',6);
            pin.parse(varargin{:});
            p = pin.Results;
            
            if ~isempty(p.timeFlags) && (~isempty(p.from) || ~isempty(p.from))
                error('Use from and to OR flags but not both');
            end
            
            if isempty(p.timeFlags) && isempty(p.from)
                p.from = e.time(1);
            end
            
            if isempty(p.timeFlags) && isempty(p.to)
                p.to = e.time(end);
            end           
            
            %Retrieve the spike counts on the training trials and training times
            if isempty(p.timeFlags)
                theseTimes = e.time >= p.from & e.time <= p.to;
            else
                theseTimes = p.timeFlags;
            end
            
            counts = cellfun(@(cts) reshape(cts(:,theseTimes),1,[]),e.spkCounts,'uniformoutput',false);
            
            %Renumber/pool conditions
            newConds = unique(p.reassignedConds);
            nConds = numel(newConds);
            for i=1:nConds;
                %Pool the spike counts across the requested conditions
                theseConds = p.reassignedConds==newConds(i);
                countsPooled{i} = cell2mat(counts(theseConds)');
                
                %Mean eye position per condition for this window (i.e., averaged over time)
                eyeX(i) = mean(reshape(e.ex(theseConds,theseTimes),1,[]));
                eyeY(i) = mean(reshape(e.ey(theseConds,theseTimes),1,[]));
            end
            eyeX = eyeX';
            eyeY = eyeY';
            countsPooled = countsPooled';
            
            %Fit the EPF using a GLM with regression function/predictors as stored in the expert object
            f = epfGLM(pin.Unmatched);
            f = glm(f,eyeX,eyeY,countsPooled);
            
            %Mean spike count per condition for this window
            countMean = cellfun(@mean,countsPooled);
            countSTE = cellfun(@(x) std(x)./sqrt(numel(x)),countsPooled);
            
            %Plot the EPF
            if p.plot
                if isempty(p.fig_h)
                    p.fig_h = figure;
                end
                figure(p.fig_h);
                
                %Plot the fitted eye-position field using a polar surface
                nRadii = 10;
                th = linspace(-pi,pi,2*nRadii);
                rad = linspace(0, max(eyeX)*1.2, nRadii);
                [thGrid,radGrid] = meshgrid(th,rad);
                [xGrid,yGrid]=pol2cart(thGrid,radGrid);
                surface = f.feval(xGrid,yGrid);
                surf(xGrid,yGrid,surface,'facealpha',0.8); hold on;
                set(gca,'xtick',[min(xlim),0,max(xlim)],'ytick',[min(ylim),0,max(ylim)],'ztick',zlim);
                
                %Plot the data points
                h = reflinexyz(eyeX,eyeY,countMean,'linestyle','-','color',[0.5 0.5 0.5],'linewidth',2); delete(h([1 2]));
                h = errorbar3(eyeX,eyeY,countMean,countSTE,'o');
                set(h,'markersize',20,'markerfacecolor','r','markeredgecolor','k');
                zlabel('Mean spike count (lambda)'); grid on;
                %Plot markers on the floor
                nPoints = 1000;
                rad = diff(ylim)*0.05;
                [circC,circY]=pol2cart(linspace(-pi,pi,nPoints),ones(1,nPoints)*rad);
                for i=1:numel(eyeX)
                    h = patch(eyeX(i)+circC,eyeY(i)+circY,ones(1,nPoints).*min(zlim),'k');
                    set(h,'facecolor',[0.8 0.8 0.8]);
                end
                
                xlabel('Eye position (hor)'); ylabel('Eye position (ver)');
            end
            
            out.epf = f;
            out.mean = countMean;
            out.ste = countSTE;
            out.ex = eyeX;
            out.ey = eyeY;
            out.r2 = 1-sum((f.feval(eyeX,eyeY)-countMean).^2)/sum(countMean-mean(countMean)).^2;
            fig_h=p.fig_h;
        end
        
        function entr = entropy(e,counts)
            
            %Returns the entropy of an expert's likelihood function [p(eye|count)] for a
            %given spike count(s)
            if ~exist('counts','var')
                counts = e.minSpikeCount:e.maxSpikeCount; %default to all observed spike counts
            end
            
            %Calculate entropy for each of the spike counts in the test set.
            prob = double(probEyeGivenCount(e,counts));
            
            %Convert from log(prob) to prob
            if e.probAsLog
                entr = -sum(exp(prob).*prob,1);
            else
                entr = -sum(prob.*log(prob),1);
            end
        end
        
        function entr = entropyPerTrial(e)
            
            %Calculate the entropy values for every spike count in the test trials/samples
            
            %For speed, entropy is calculated once for each of the possible spike counts and then allocated using spike counts as indices.
            
            %Get the unique counts and entropies
            uniqueCounts = e.minSpikeCount:e.maxSpikeCount;
            entr = entropy(e,uniqueCounts);
            
            %Get the spike counts and look up entropies
            counts = cellfun(@(a,b) a(b,e.testSet.times), e.spkCounts,e.testSet.trials,'uniformoutput',false);
            count2entr = @(cts) entr(c2i(e,cts)); %Use spike counts as indices into base-1 vector
            entr = cellfun(count2entr,counts,'uniformoutput',false);
            
        end
        
        function nConditions = get.nConditions(e)
            nConditions = numel(e.spkCounts);
        end
        
        function ind=c2i(e,count)
            %Convert a spike count into an index into e.pEyeGivenCount
            ind = count-e.minSpikeCount+1;
        end
    end
end