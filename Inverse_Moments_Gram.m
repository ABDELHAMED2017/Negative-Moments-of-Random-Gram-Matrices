%**************************************************************************************************%
%                                     Inverse_Moments_Gram
%
% This function implements the inverse moments derived in "Analytical Derivation
% of the Inverse Moments of One-Sided Correlated Gram Matrices With Applications",
% Khalil Elkhalil, Abla Kammoun, Tareq Y. Al-Naffouri and Mohamed-Slim Alouini
% IEEE Transactions on Signal Processing, 2016.
% This function exactly computes the following quantity:
% moment = trace[(H'*Theta*H)^(-r)], 1 <= r <= min(m,n-m).
% where Theta is a positive definite matrix of dimension n x n with
% distinct eigenvalues (this is very important, otherwise the function will
% not work), theta_1 > theta_2 >...> theta_n. The matrix H is a (n x m) complex
% Gaussian random matrix with i.i.d zero mean unit variance entries.
% Inputs : 
            % Theta = The correlation matrix
            % r = The moment order
            % m = The size of the Gram matrix H'*Theta*H.
% Outputs :  moment           
% Coded by: Khalil Elkhalil and Abla Kammoun, KAUST, Saudi Arabia.
% E-mails: khalil.elkhalil@kaust.edu.sa, abla.kammoun@kaust.edu.sa,
%         tareq.alnaffouri@kaust.edu.sa, slim.alouini@kaust.edu.sa.
% Copyright (c) Khalil Elkhalil, Abla Kammoun, Tareq Y. Al-Naffouri and Mohamed-Slim Alouini, 2015
%
%               Computer, Electrical, Mathematical Sciences and Enginnering (CEMSE),
%               King Abdullah University of Science and Technology (KAUST).
%
% **************************************************************************************************%
function moment = Inverse_Moments_Gram(Theta,r,m)
[~,n] = size(Theta) ;
[~,pos_def] = chol(Theta) ;
Theta_eigenvalues = eig(Theta) ;
% Check if the input matrix Theta is positive definite with distinct
% eigenvalues
distinct_eig = unique(Theta_eigenvalues) ;
if length(distinct_eig) < n || pos_def ~= 0
    error(message('The correlation matrix Theta is either not positive definite or has equal eigenvalues'));
end
%%%%%%% Computation Matrix Psi %%%%%
matrixPsi = [Theta_eigenvalues(1:n)]*ones(1,n);
matrixpower = ones(n,1)*[0:1:n-1] ;
Psi_tot = matrixPsi.^matrixpower ;
Psi = Psi_tot(1:n-m,1:n-m) ;

% Conditioning
c = 0 ;
Psi = Psi+c*eye(size(Psi)) ;
Psi_sub = Psi_tot(n-m+1:end,1:n-m) ;

%%% Computation matrix D %%%%
for k = 1:m
    factor = factorial(k-1);
    for l=1:m
        vec_theta_g = Theta_eigenvalues(n-m+l).^([0:1:n-m-1]') ;
        vec_theta_d = Theta_eigenvalues(1:n-m).^([n-m+k-1]) ;
        Dmatrix(l,k) = factor*(Theta_eigenvalues(n-m+l)^(n-m+k-1)-vec_theta_g'*inv(Psi)*vec_theta_d) ;
    end
end

D = cof(Dmatrix) ;
csi_tab = [0.001:0.05:20] ;

product = 1 ;
for ii = 2:n
    product = product*prod(Theta_eigenvalues(ii:end)-Theta_eigenvalues(ii-1)) ;
end
product_fac = 1 ;
for ii = 1:m-1
    product_fac = product_fac*factorial(ii) ;
end
L = det(Psi)/(m*product_fac*product) ;

%%%% Theoretical computation of the variance %%%%%
variance_vec = 0 ;
variance_matrix = 0 ;
v = Theta_eigenvalues(1:n-m).^(n-m-1) ;
J = Psi.*(log(Theta_eigenvalues(1:n-m))*ones(1,n-m)) ;
for ii = 1:m
    theta_i = Theta_eigenvalues(n-m+ii).^[0:1:n-m-1]' ;
    ji = log(Theta_eigenvalues(n-m+ii))*theta_i ;
    for jj = 1:r
        Omega_j = Theta_eigenvalues(1:n-m).^(n-m-r+jj-1) ;
        variance_vec = variance_vec+D(ii,jj)*(-1)^(r-jj)/(factorial(r-jj))*ji'*inv(Psi)*Omega_j ;
        variance_matrix = variance_matrix+D(ii,jj)*(-1)^(r+1-jj)/(factorial(r-jj))*theta_i'*inv(Psi)*J*inv(Psi)*Omega_j ;
    end
end
moment = L*(variance_vec+variance_matrix) ;
end