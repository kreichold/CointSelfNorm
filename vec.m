function ystacked = vec(y)
% stack the columns of y
ystacked = reshape(y,numel(y),1); % = y(:);
end