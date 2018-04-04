function out = statistics(time,decX,decY,ex,ey,ent,xSpace,ySpace)
%Summary statistics for the output of a decoding run

%Calculate the mean, std, median, 25th percentile, and 75th percentile at each timepoint for each condition
out.X = stats(decX);
out.Y = stats(decY);

%Add the 2D mode (i.e. most decoded position) to the stats structure
for i=1:numel(decX)
    for j=1:size(decX{i},1)
        n = hist3([decX{i}(j,:)',decY{i}(j,:)'],{xSpace',ySpace'});
        [~,ind]=max(n(:));
        [maxi,maxj]=ind2sub(size(n),ind);
        out.X.mode(j,i) = xSpace(maxi);
        out.Y.mode(j,i) = ySpace(maxj);
    end
end

%Calculate SCATTER - i.e. the Euclidean distance of each decoded position to the median [x,y]
dim = ndims(decX{1});
nSamples = size(decX{1},dim);
    %Deviations in one-dimenion (hor or ver)
dev = @(vals) vals-repmat(median(vals,dim),1,nSamples);
    %Deviations in 2D (Euclidean distance)
deviations = @(xVals,yVals) hypot(dev(xVals),dev(yVals));
nConds = numel(decX);
nTimes = size(decX{1},1);
for i=1:nTimes
    for j=1:nConds
        %Record the scatter as the median deviation
        out.scatter(:,j) = median(deviations(decX{j},decY{j}),dim);
        
        %Record the covariance matrix (used for plotting error ellipses later)
        out.covXY(i,j,:,:) = cov([decX{j}(i,:);decY{j}(i,:)]');
    end
end

%Accuracy measures relative to the TRUE eye positions
out.constError.ofMean = accuracy(out.X.mean,out.Y.mean,ex,ey);
out.constError.ofMedian = accuracy(out.X.p50,out.Y.p50,ex,ey);
out.constError.ofMode = accuracy(out.X.mode,out.Y.mode,ex,ey);

%Record the time and true eye positions
out.time = time;
out.ex = ex;
out.ey = ey;

if ~isempty(ent)
    out.entropy = stats(ent);
end

function out = accuracy(x,y,ex,ey)

%Calculate constant error (dx,dy,and Euclidean distance)
out.X = x - ex;
out.Y = y - ey;
out.dist = hypot(x - ex,y - ey);

%How well does the decoder capture variability in the true eye over CONDITIONS at each time point?
out.sse.perTime = sum(out.dist.^2,2);
out.sst.perTime = sum(hypot(ex - repmat(mean(ex,2),1,size(ex,2)),ey - repmat(mean(ey,2),1,size(ey,2))).^2,2); %Distance of each eye position to the centroid of all eye positions
out.r2.perTime = 1-(out.sse.perTime./out.sst.perTime);

%How well does the decoder capture variability in the true eye over TIME in each condition?
out.sse.perCond = sum(out.dist.^2,1);
out.sst.perCond = sum(hypot(ex - repmat(mean(ex,1),size(ex,1),1),ey - repmat(mean(ey,1),size(ey,1),1)).^2,1); %Distance of each eye position to the centroid of all eye positions
out.r2.perCond = 1-(out.sse.perCond./out.sst.perCond);

function out = stats(data)

dim = ndims(data{1});
for i=1:numel(data)
    out.mean(:,i) = mean(data{i},dim);
    out.std(:,i) = std(data{i},[],dim);
    out.p25(:,i) = prctile(data{i},25,dim);
    out.p50(:,i) = prctile(data{i},50,dim);
    out.p75(:,i) = prctile(data{i},75,dim);
end