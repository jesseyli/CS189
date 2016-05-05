% Part (a)
mu = [1 1];
Sigma = [2 0;0 1];
x1 = -3:.2:5; x2 = -2:.2:4;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
figure()
contour(X1,X2,F)
set(gca,'FontSize', 14)
title('part a')
xlabel('X1')
ylabel('X2')

% Part (b)
mu = [-1 2];
Sigma = [3 1;1 2];
x1 = -5:.2:3; x2 = -2:.2:5;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
figure()
contour(X1,X2,F)
set(gca,'FontSize', 14)
title('part b')
xlabel('X1')
ylabel('X2')

% Part (c)
mu1 = [0 2];
Sigma1 = [1 1;1 2];
mu2 = [2 0];
Sigma2 = [1 1;1 2];
x1 = -6:.2:6; x2 = -6:.2:6;
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],mu1,Sigma1);
F2 = mvnpdf([X1(:) X2(:)],mu2,Sigma2);
F3 = F1 - F2;
F3 = reshape(F3,length(x2),length(x1));
figure()
contour(X1,X2,F3,10)
set(gca,'FontSize', 14)
title('part c')
xlabel('X1')
ylabel('X2')

% Part (d)
mu1 = [0 2];
Sigma1 = [1 1;1 2];
mu2 = [2 0];
Sigma2 = [3 1;1 2];
x1 = -6:.2:6; x2 = -6:.2:6;
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],mu1,Sigma1);
F2 = mvnpdf([X1(:) X2(:)],mu2,Sigma2);
F3 = F1 - F2;
F3 = reshape(F3,length(x2),length(x1));
figure()
contour(X1,X2,F3,10)
set(gca,'FontSize', 14)
title('part d')
xlabel('X1')
ylabel('X2')

% Part (e)
mu1 = [1 1];
Sigma1 = [1 0;0 2];
mu2 = [-1 -1];
Sigma2 = [2 1;1 2];
x1 = -7:.2:7; x2 = -7:.2:7;
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],mu1,Sigma1);
F2 = mvnpdf([X1(:) X2(:)],mu2,Sigma2);
F3 = F1 - F2;
F3 = reshape(F3,length(x2),length(x1));
figure()
contour(X1,X2,F3,10)
set(gca,'FontSize', 14)
title('part e')
xlabel('X1')
ylabel('X2')