% Andrew North - 10/11/2021
% System Identification using the Least Squares method
% Given x y data (position vs time) or (acceleration vs time)
% Want to find the parameters for the governing equation (could be a first
% or second order differential equation

phi = [0.0 1; 0.5 1; 1.0 1; 1.5 1; 2.0 1; 2.5 1; 3.0 1]
Y = [1.6501; 1.1311; 1.7068; 1.7860; 2.3913; 2.4621; 2.3565]
x = [0:0.5:3]

plot(x,Y,'*')

% Phi transpose 
phiT = (phi)'

% ThetaLS is the 'Parameter Vector' solved using the Least Squares method
thetaLS = inv(phiT*phi)*phiT*Y

% Best fit line using least squares method
m = thetaLS(1,1)
b = thetaLS(2,1)

xx = [0,0.05,3]
yy = m*xx+b
plot(x,Y,'*',xx,yy)
ylim([0 3])

