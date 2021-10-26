% ME520 System ID Assignment

close all; clear; clc;

% ------------------------- STEP 2 ------------------------- %
M = readmatrix('data_SystemID.csv');

T = M(1:end-2,1)'; % time matrix
Usys = M(1:end-2,2)'; % input matrix

Y = M(1:end-2,3);  % Y = third column of M
phi_1 = -M(2:end-1,3);  % -y(k-1)  output
phi_2 = -M(3:end,3);    % -y(k-2)  output
phi_3 = M(2:end-1,2);   % u(k-1)   input
phi_4 = M(3:end,2);     % u(k-2)   input
phi = [phi_1 phi_2 phi_3 phi_4];
phiT = (phi)';

thetaLS = inv(phiT*phi)*phiT*Y;

% calculate the parameters
a1 = thetaLS(1,1);
a2 = thetaLS(2,1);
b1 = thetaLS(3,1);
b2 = thetaLS(4,1);

% Test the new 2nd order system model:
y_1 = a1*phi_1;
y_2 = a2*phi_2;
y_3 = b1*phi_3;
y_4 = b2*phi_4;
y_test = [y_1+y_2+y_3+y_4];
% Or cleaner implementation:
y_hat = phi*thetaLS;

% Plot both the original input and output data, and the output from the new
% system model
figure;
plot(T,Usys,':',T,Y,':',T,y_test,'LineWidth',2); 
title('System Model Test')
xlabel('time (seconds)')
ylabel('volts')
legend('System Input','System Output','Simulated Output')

% Check the error with residuals
e = Y-y_hat;
figure;
plot(T,e,'-*');
title('2nd Order System Model Residuals (Error)');
xlabel('time (seconds)');
ylabel('Error (voltage)');

% % Try a 1st order system
% phi = [phi_1 phi_3];
% thetaLS = ((phi'*phi)^-1)*phi'*Y;
% a1 = thetaLS(1,1);
% b1 = thetaLS(2,1);
% % Test the 1st order system model:
% y_hat = phi*thetaLS;
% e = Y-y_hat;  % Check residual errors
% plot(T,e,'-*')
% title('1st Order System Model Residuals (Error)');
% xlabel('time (seconds)');
% ylabel('Error (voltage)');

% % Try a 3rd order system
% phi_1 = -M(2:end-2,3); % -y(k-1)
% phi_2 = -M(3:end-1,3); % -y(k-2)
% phi_3 = -M(4:end,3);   % -y(k-3)
% phi_4 = M(2:end-2,2);  % u(k-1) 
% phi_5 = M(3:end-1,2);  % u(k-2)
% phi_6 = M(4:end,2);    % u(k-3)
% phi = [phi_1 phi_2 phi_3 phi_4 phi_5 phi_6];
% Y = M(1:end-3,3);  % re-define Y length
% T = M(1:end-3,1);  % re-define time matrix length
% thetaLS = ((phi'*phi)^-1)*phi'*Y;  % Parameters vector
% y_hat = phi*thetaLS;
% e = Y-y_hat;  % Check residual errors
% plot(T,e,'-*');
% title('3rd Order System Model Residuals (Error)');
% xlabel('time (seconds)');
% ylabel('Error (voltage)');

% ------------------------- STEP 3 ------------------------- %
% Transfer function sys
sys = tf([b1,b2],[1,a1,a2]);
U = M(2:end,2);  % U is the input data
Y = M(2:end,3);
L = length(U);
Fs = 50;    % Hz


% Input frequency domain ID using fft
fft_inp = fft(U);
P2 = abs(fft_inp/L);
P1 = P2(1:L/2+1);  % graph on y-axis
P1(2:end-1) = 2*P1(2:end-1);
freq_inp = Fs*(0:(L/2))/L;  % graph on x-axis

% Output frequency domain ID using fft
fft_out = fft(Y);
P2o = abs(fft_out/L);
P1o = P2o(1:L/2+1);  % graph on y-axis
P1o(2:end-1) = 2*P1o(2:end-1);
freq_out = Fs*(0:(L/2))/L;  % graph on x-axis

% System frequency response
fft_G = fft_out./fft_inp;
P2g = abs(fft_G/L);
P1g = P2g(1:L/2+1);  % graph on y-axis
P1g(2:end-1) = 2*P1g(2:end-1);
freq_G = Fs*(0:(L/2))/L;  % graph on x-axis

% Plot frequency response
figure;
plot(freq_G,P1g);
title('System Frequency Response');
xlabel('frequency (Hz)')
ylabel('Magnitude')

% ------------------------- STEP 4 ------------------------- %

