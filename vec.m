function ystacked = vec(y)
% stacks the columns of y
ystacked = reshape(y,numel(y),1); % = y(:);
end