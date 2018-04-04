classdef epfGLM
    %Generic class to fit a 2D regression (GLM) EPF to neural spike counts.
    %Currently used only by the @expert class
    properties (SetAccess=public)
        predFn
        type
        minVal;
    end
    
    properties (Dependent=true)
        pVal
        inModel
    end
    
    properties (Access=public)
        mdl
    end
    
    methods (Access=private)

    end

    methods
        
        function f=epfGLM(varargin)       
            %GLM model for an eye position field.
            pin=inputParser;
            pin.addParameter('glmType','FULL',@(x) any(strcmpi({'FULL','STEPWISE'},x)));
            pin.addParameter('predFn',@(x,y) [x, y, x.^2, y.^2, x.*y]);
            pin.addParameter('minVal',0.25,@isnumeric);
            pin.parse(varargin{:});
            f.type = pin.Results.glmType;
            f.predFn = pin.Results.predFn;
            f.minVal = pin.Results.minVal;
        end
        
        function f = glm(f,ex,ey,counts)
            
            %Fit a regression surface across eye positions
            mleFun = @(cnt) mle(cnt,'distribution','POISSON');
            anchorPrms = cellfun(mleFun,counts);
            switch f.type
                case 'FULL'
                    f.mdl = fitglm(f.predFn(ex,ey),anchorPrms);
                case 'STEPWISE'
                    %Perform stepwsie regression.
                    %On rare occasions, gets stuck in infinite loop (bug in
                    %Matlab's stepwiseglm?). Try to find a solution up to 5
                    %times, otherwise, throw error
                    maxSteps = 100;
                    found=false;
                    for i=1:5
                        f.mdl = stepwiseglm(f.predFn(ex,ey),anchorPrms,'constant','verbose',0,'nSteps',maxSteps);
                        if size(f.mdl.Steps.History,1)<maxSteps
                            found = true;
                            break;
                        else
                            %Make a micro change to the data, to break the deadlock
                            anchorPrms = anchorPrms + (rand(size(anchorPrms))-0.5)*0.00001;
                        end
                    end
                    if ~found
                        error('Stepwise stuck in an infinite loop');
                    end
            end
        end
        
        function c = feval(f,ex,ey)
            %Evaluate points on the surface at [ex,ey]
            
            if ~isequal(size(ex),size(ey))
                error('ex and ey must be the same size');
            end

            %Calculations are all done in 1D, so linearlise inputs arguemtns and return to original size after
            origSize = size(ex);
            ex = ex(:);
            ey = ey(:);
            
            %Create the regressor matrix
            predictors = f.predFn(ex,ey);
            
            %Evaluate the EPF function at the points (using only the terms that are actually in the model)
            c = f.mdl.feval(predictors(:,f.mdl.VariableInfo.InModel));
            
            %Flood the output to the specified minimum value
            c(c<f.minVal) = f.minVal;
            
            %Restore original dimensionality
            c=reshape(c,origSize);
        end    
        
        function p = get.pVal(f)
            p = f.mdl.coefTest;
        end 
        
        function isIn = get.inModel(f)
            %Boolean vector indicating which predictors are included in the fitted model.
            isIn = f.mdl.VariableInfo.InModel;
            
            %The InModel vector includes a term for the dependent variable (y). We don't care about that, so remove it.
            isIn(strcmpi(f.mdl.VariableNames,'y'))=[];
        end
        
        function h = isSig(f,alpha)
            %Is the overall GLM significantly different from the constant-onl model?
            if ~exist('alpha','var')
                alpha = 0.05;
            end
            h = f.pVal<alpha & any(f.inModel);
        end
    end
end