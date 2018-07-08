% LEGENDRE_DERIVATIVE    Compute first derivative of (normalized)
%                        associated Legendre polynomial
%
% Computing accurate derivatives of associated Legendre polynomials can
% be tricky. Even in advanced texts, they are usually written as recurrence
% relations and/or with (normalization) factors involving factorials.
%
% A naive implementation will therefore quickly run into the limits of
% IEEE754 double-precision, resulting in NaN/inf or significant loss of
% precision, already at relatively low degree N.
%
% LEGENDRE_DERIVATIVE is a fully vectorized, numerically stable and
% robustly validated implementation of the derivative computation. It
% allows fast and accurate computations of the derivatives for any degree N.
%
% LEGENDRE_DERIVATIVE works the same as MATLAB's own LEGENDRE, except it
% does not compute the polynomial values, but the values of the derivatives.
%
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
%            X : array of points at which to evaluate the derivatives.
%          Pnm : values of the Legendre polynomial at X; these are computed
%                automatically when omitted. Since the derivatives only
%                depend on the degree/order N,M and the values of the
%                polynomials Pnm(x), providing these values if they are
%                already computed elsewhere will thus improve performance.
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
    % https://www.paypal.me/RodyO/3.5
    
    %% Initialization 

    % Parse input, do some checks
    if verLessThan('MATLAB', '7.3')
        error(nargchk(2, 4, nargin, 'struct')); %#ok<*NCHKN>
    else
        narginchk(2,4);
    end

    n = varargin{1};
    varargin = varargin(2:end);
    assert(isnumeric(n) && isscalar(n) && isfinite(n) && isreal(n) && round(n)==n && n>=0,...
          [mfilename ':invalid_n'],...
          'Degree N must be a positive integer.');

    normalization = 'unnorm';
    if ischar(varargin{end})
        normalization = varargin{end};
        varargin(end) = [];
    end

    assert(any(strcmpi(normalization, {'unnorm','norm','sch'})),...
          [mfilename ':unsupported_normalization'],...
          'Unsupported normalization specified: ''%s''.', normalization);

    x = varargin{end};
    varargin(end) = [];
    
    assert(~isempty(x) && isnumeric(x) && ...
           all(isreal(x(:))) && all(abs(x(~isnan(x(:))))) <= 1,...
           [mfilename ':invalid_x'],...
           'X must be real, numeric, nonempty, and in the range (-1,1).');
    
    if isempty(varargin)
        Pnm = legendre(n,x,normalization);
    else
        Pnm = varargin{end};
    end
    
    assert(size(Pnm,1)==n+1,...
          [mfilename  ':Pnm_dimension_mismatch'],...
          'Dimensions of polynomial values Pnm disagrees with degree N.');
      
    % Special case; MATLAB does not normally reshape the Pnm vector 
    % for size(x) = 1�p  
    szPnm = size(Pnm);    
    szX   = size(x);
    ndx   = ndims(x);
    assert(isscalar(x) && szPnm(2)==1 || ...
           isequal(szX(1:find(szX ~= 1, 1, 'last')), szPnm(2:end)) || ...
           (szX(1) == 1 && ndx == 2 && szX(2) ~= 1 && szPnm(2)==szX(2)),...
           [mfilename  ':Pnm_dimension_mismatch'],...
           'Dimensions of polynomial values Pnm disagrees with vector x.');
                     
    %% Computation 
    
    % n==0 case
    if numel(Pnm)==1
        dPnmdx = zeros(size(x)); 
        return; 
    end    

    % Initialize some arrays for vectorization    
    x   = permute(x, [ndx+1 1:ndx]);
    idx = repmat({':'}, ndx,1);
    m   = (0:n).';
    sqx = 1 - x.^2;
    
    % Special case; MATLAB does not normally reshape the Pnm vector 
    % for size(x) = 1�p     
    if szX(1)==1 && ndx == 2
        Pnm = permute(Pnm, [1 3 2]); end
    
    % Normalization factors: this is actually a nice puzzle :)
    F = -ones(n+1,1);
    if ~strcmpi(normalization,'unnorm')

        % Factors for both normalizations are the same...
        s = 1/n/(n+1);
        for ii = m.'
            F(ii+1) = s;
            s = s * (n-ii+1)/(n+ii+1)*(n+ii)/(n-ii);
        end

        %... save for the first few entries
        if numel(F)>1
            switch normalization
                case 'norm'
                    F(1) = 1/F(2);
                case 'sch'
                    F(1) = 1/2/F(2);
                    F(2) = 1/F(1);
            end
        end

        F = sqrt(F);

    end    
    
    % Compute derivative, vectorized, efficient.
    % Sadly, that means it's unreadable.
    dPnmdx = bsxfun(@rdivide, Pnm .* bsxfun(@times, m, x) - ...
                                     bsxfun(@times, F, ...
                                            [-Pnm(2,idx{:})/n/(n+1)
                                            +Pnm(1:end-1,idx{:})]...
                                     ) .*  ...
                              bsxfun(@times, (n+m).*(n-m+1), sqrt(sqx)), ...
                    sqx);
                
    % Handle edge cases
	ispolar = abs(x)==1;
    if any(ispolar)
        
        xp  = x(ispolar);
        pwr = +xp.^(1+n);
        
        dPnmdx(4:end,ispolar) = 0;
        
        switch normalization
            case 'norm'   
                dPnmdx(1,ispolar) = +pwr .* Pnm(1,:) .* n*(n+1)/2;
                dPnmdx(2,ispolar) = -xp.*pwr*inf;
                % TODO: 3 = "engineered", not actually derived...
                if n > 1
                    dPnmdx(3,ispolar) = -pwr.* sqrt(n*(n+1)/(F(3)+4))*abs(Pnm(1))*F(1)/2; end
                
            case 'sch'
                dPnmdx(1,ispolar) = +pwr .* n*(n+1)/2;
                dPnmdx(2,ispolar) = -xp.*pwr*inf;
                if n > 1
                    dPnmdx(3,ispolar) = -pwr .* sqrt(n*(n+1)/8)/F(3); end
                                
            otherwise
                dPnmdx(1,ispolar) = +pwr .* n*(n+1)/2;                
                dPnmdx(2,ispolar) = +xp.*pwr*inf;       
                if n > 1
                    dPnmdx(3,ispolar) = -pwr * (n-1)*n*(n+1)*(n+2)/4; end
        end
    end
    
end
