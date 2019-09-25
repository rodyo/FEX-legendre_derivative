classdef (TestTags = {'Legendre' 'Derivatives' 'Unit test'})...
         TestLegendreDerivative < matlab.unittest.TestCase
     
    %% Test harness =======================================================
       
    properties (Constant)        
        num_random_trials   = 2e2  % number of random x's to use
        max_n               = 30   % maximum order to try
        h_finite_difference = 1e-5 % starting step for finite differences 
        tolerance           = 1e-2 % relative error. Yes, this is rather
                                   % high, but it's insanely difficult to 
                                   % compute accurate derivatives near the 
                                   % edges via finite differences.
    end
        
    properties (TestParameter)
        
        % Test parameters edge cases        
        edges = {-1 +1}
        
        % The different normalisations
        norms = {'unnorm' 'norm' 'sch'}        
        
    end
      
    %% Test cases =========================================================
    
    methods (Test,...
             TestTags = {'Interface tests'})
        
        % Invalid argument count
        function TestInvalidNargin(tst)
            tst.verifyError(@() legendre_derivative(0,0,0,0,0),...
                            'MATLAB:narginchk:tooManyInputs');
            tst.verifyError(@() legendre_derivative(),...
                            'MATLAB:narginchk:notEnoughInputs')
        end
        
        % Invalid order
        function TestInvalidN(tst, norms)
            
            eid = 'legendre_derivative:invalid_n';
            f   = @(n) @() legendre_derivative(n, 0, norms);
            
            % There's a number of conditions that should trigger this
            % error. First, n is simply out-of-bounds: 
            tst.verifyError(f(-1), eid);   
            
            % When X is not numeric, empty, or some other funky thing:
            tst.verifyError(f(1.1), eid);   tst.verifyError(f([1 2]), eid);
            tst.verifyError(f([]), eid);    tst.verifyError(f(-1i), eid);
            tst.verifyError(f(inf), eid);   tst.verifyError(f(+1i), eid);
            tst.verifyError(f(@cos), eid);  tst.verifyError(f('nop'), eid);
            
        end
         
        % Invalid value for X
        function TestInvalidX(tst, norms) 
             
            n   = 7;
            eid = 'legendre_derivative:invalid_x';
            f   = @(x) @() legendre_derivative(n, x, norms);
            
            % There's a number of conditions that should trigger this
            % error. First, x is simply out-of-bounds: 
            tst.verifyError(f(-15), eid);   tst.verifyError(f(+15), eid);
            
            % When X is not numeric, empty, or some other funky thing:
            tst.verifyError(f([]), eid);    tst.verifyError(f(-1i), eid);
            tst.verifyError(f(inf), eid);   tst.verifyError(f(+1i), eid);
            tst.verifyError(f(@cos), eid);  tst.verifyError(f('nop'), eid);
            
        end
        
        % Invalid normalisation 
        function TestInvalidNormalisation(tst)
            tst.verifyError(@() legendre_derivative(0,0, 'invalid_norm'),...
                            'legendre_derivative:unsupported_normalization');
        end
        
        % Pnm dimensions mismatch
        function TestPnmDimsError(tst, norms)  
            
            n   = 7;
            x   = rand;                        
            f   = @(Pnm) @()legendre_derivative(n, Pnm, x, norms);
            Pnm = legendre(n,x,norms);
            eid = 'legendre_derivative:Pnm_dimension_mismatch';
            
            tst.verifyError(f([Pnm Pnm])   , eid);
            tst.verifyError(f([Pnm;  3])   , eid);
            tst.verifyError(f(Pnm(1:end-1)), eid);
            
        end
         
    end
    
    methods (Test,...
             TestTags = {'Functional tests'})
         
        % Test whether the outcomes are equal when calling the function 
        % with and without argument 'Pnm' 
        function TestPnmArg(tst, norms)
             
            % Just take a random value for x & n
            n = randi(tst.max_n);
            x = 2*rand - 1;
            
            D1 = legendre_derivative(n, x, norms);
            
            Pnm = legendre(n, x, norms);
            D2  = legendre_derivative(n, Pnm, x, norms);
            
            tst.verifyTrue(tst.are_equal(D1,D2), [... 
                           'Using argument ''Pnm'' changes the values ',...
                           'beyond reasonable tolerance.']);
        end
                 
        % Test whether legendre_derivative() can produce the same results 
        % as finite differences on legendre() for all types of 
        % normalisations using a bunch of random values and random orders
        function TestRandomValues(tst, norms)
            
            h = tst.h_finite_difference;
            n = randi(tst.max_n, tst.num_random_trials, 1);
            x = (1-h) * (2*rand(tst.num_random_trials, 1) - 1);             
            
            for ii = 1:tst.num_random_trials
                
                % Value to check 
                D = legendre_derivative(n(ii), x(ii), norms);
                
                % Central differences with Richardson extrapolation 
                d = tst.richardson_derivative('ctr', x(ii), n(ii), norms);
                                                
                % Check if they are equal to within tolerance
                tst.verifyTrue(tst.are_equal(d,D), [...
                               'Values of computed derivative and ',...
                               'numerically estimated derivative differ.']);
            end
            
        end
        
        % Test n==0 case
        function TestZerothOrder(tst, norms)
            
            h = tst.h_finite_difference;
            n = 0;
            x = (1-h) * (2*rand(tst.num_random_trials, 1) - 1);             
            
            for ii = 1:tst.num_random_trials
                tst.verifyEqual(legendre_derivative(n,x(ii), norms), 0, ...
                               'Value of derivative is nonzero for n==0.');                
            end           
            
        end
        
        % Test the edge cases: x = ±1
        function TestEdgeCases(tst, edges, norms)
             
            n = 5; % <- randomly chosen, doesn't really matter
                        
            % Value to check 
            D = legendre_derivative(n, edges, norms);
            
            % Central differences with Richardson extrapolation 
            if edges < 0
                d = tst.richardson_derivative('fwd', edges, n, norms);
            else
                d = tst.richardson_derivative('bwd', edges, n, norms);
            end
                                    
            % Check if they are equal to within tolerance
            msg = ['Computed derivative and numerically ',...
                   'estimated derivative differ for edge case.'];               
               
            % The second element tends to infinity; check if that's true,
            % and then remove it for the comparison. Same for zero-valued
            % elements
            tst.verifyTrue(sign(D(2))==sign(d(2)) && ...
                           abs(D(2)) >= 1e3 && abs(d(2)) >= 1e3,...
                           msg);
                       
            tst.verifyTrue(tst.are_equal(d,D), msg);
            
        end
                 
    end
        
    % Helper methods (non-tests) 
    methods
        
        % Check for equality-with-a-relative-tolerance, in the presence of 
        % zeros or infinities
        function yn = are_equal(tst, D1,D2)
            
            e = tst.tolerance;
            b = 1e3;

            % Remove zero-valued or non-finite elements for the comparison
            zero      = (abs(D1) <= e & abs(D2) <= e);
            D1(zero)  = [];   tst.assertNotEmpty(D1);
            D2(zero)  = [];   tst.assertNotEmpty(D2);
            
            infty     = (abs(D1) >= b & abs(D2) >= b);
            D1(infty) = [];   tst.assertNotEmpty(D1);
            D2(infty) = [];   tst.assertNotEmpty(D2);      

            % Finally, the actual comparison:             
            yn = all(abs((D1-D2)./max(norm(D1),norm(D2))) <= e);
                       
        end
        
        % Compute numerical derivatives as accurately as we can. Nominally,
        % we do that with central differences + Richardson extrapolation,
        % but that doesn't work for the edge cases. There, we resort to 
        % using forward/backwards differences
        function df = richardson_derivative(tst, dftype, x, n, norm)
            
            h = tst.h_finite_difference;
                        
            switch (dftype)
                case 'fwd', N = 24; % NOTE: yes, cherry-picked...
                case 'bwd', N = 24;
                case 'ctr', N = 7;
            end
            
            fc = @(h) legendre(n, x+h, norm) - legendre(n, x-h, norm);
            ff = @(h) legendre(n, x+h, norm) - legendre(n, x  , norm);
            fb = @(h) legendre(n, x  , norm) - legendre(n, x-h, norm);
            
            d = zeros(N,N, n+1);
            for ii = 1:N
                
                switch (dftype)
                    case 'fwd', d(ii,1,:) = ff(h) / h;
                    case 'bwd', d(ii,1,:) = fb(h) / h;
                    case 'ctr', d(ii,1,:) = fc(h) / (2*h);
                end               
                
                h = h/2;
                
                for jj = 2:ii
                    d(ii,jj,:) = d(ii,jj-1,:) + ...
                                 (d(ii,jj-1,:) - d(ii-1,jj-1,:)) / (4^jj-1);
                end
                
            end
            
            df = permute(d(end,end,:), [3 1 2]);

        end
        
    end
     
end
