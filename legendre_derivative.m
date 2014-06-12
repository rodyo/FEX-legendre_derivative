% LEGENDRE_DERIVATIVE    Compute derivative of (normalized) associated
%                        Legendre polynomial
%
% Fully vectorized, numerically stable computation of the derivative of the
% associated Legendre polynomial of degree N. 
%
% USAGE: 
% ----------------
%    dPnmdx = legendre_derivative(N,X);
%    dPnmdx = legendre_derivative(N,X, normalization);
%    dPnmdx = legendre_derivative(N,Pnm,X);
%    dPnmdx = legendre_derivative(N,Pnm,X, normalization);
%
% INPUT ARGUMENTS: 
% ----------------
%
%            N : degree of the Legendre polynomial.
%            X : array of points at which to evaluate the derivative.
%          Pnm : values of the Legendre polynomial at X; this is computed
%                automatically when omitted.
% normalization: type of normalization to use. Can be equal to 'unnorm' 
%                (default), 'norm' (fully normalized) or 'sch' (Schmidt 
%                semi-normalized). 
%
% OUTPUT ARGUMENTS: 
% ----------------
% dPnmdx : value(s) of the derivative of the n-th order associated Legendre 
%          polynomial at all location(s) X, for all degrees M=0..N. The array 
%          dPnmdx has one more dimension than x; each element 
%          dPnmdx(M+1,i,j,k,...) contains the associated Legendre function of 
%          degree N and order M evaluated at X(i,j,k,...).
%
% See also legendre.
function dPnmdx = legendre_derivative(varargin)
 
    % If you find this work useful, please consider a donation:
    % https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=6G3S5UYM7HJ3N
    
    % Parse input, do some checks
    error(nargchk(2,4,nargin,'struct'));
    
    n = varargin{1};
    varargin = varargin(2:end);
    assert(isnumeric(n) && isscalar(n) && isfinite(n) && isreal(n) && round(n)==n && n>0,...
        'legendre_derivative:invalid_n',...
        'Degree must be a positive integer.');
    
    normalization = 'unnorm';
    if ischar(varargin{end})
        normalization = varargin{end}; varargin = varargin(1:end-1); end  
    assert(any(strcmpi(normalization, {'unnorm','norm','sch'})),...
        'legendre_derivative:invalid_normalization',...
        'Unsupported normalization type specified: ''%s''.', normalization);
    
    x = varargin{end};  
    varargin = varargin(1:end-1);
    if isempty(varargin)
        Pnm = legendre(n,x,normalization);
    else
        Pnm = varargin{end};
    end
    assert(size(Pnm,1)==n,...
        'legendre_derivative:invalid_Pnm',...
        'Dimensions of polynomial values disagrees with degree N.');
    
    % Initialize some arrays for vectorization
    x   = permute(x, [ndims(x)+1 1:ndims(x)]);
    idx = repmat({':'}, ndims(x)-1,1);
    m   = (0:n).';
    sqx = 1 - x.^2;
    
    % Normalization factors: that was a nice puzzle :) 
    F = -ones(n+1,1);
    if ~strcmpi(normalization,'unnorm')
        
        s = 1/n/(n+1);
        for m = 0:n
            F(m+1) = s;
            s = s * (n-m+1)/(n+m+1)*(n+m)/(n-m);
        end
        
        switch normalization
            case 'norm'                
                F(1) = 1/F(2);
            case 'sch'                
                F(1) = 1/2/F(2);
                F(2) = 1/F(1);
        end
        F = sqrt(F);
        
    end
        
    % Compute derivative, vectorized
    dPnmdx = bsxfun(@rdivide, ...
        Pnm .* bsxfun(@times,m,x) - ...
            bsxfun(@times, F, ...
            [-Pnm(2,idx{:})/n/(n+1)
            +Pnm(1:end-1,idx{:})]) .*  ...
            bsxfun(@times, (n+m).*(n-m+1), sqrt(sqx)), ...
        sqx);
    
end
