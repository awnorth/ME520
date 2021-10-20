% ME520 System ID Assignment

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

% Test the transfer function with the difference equation:
y_1 = a1*phi_1;
y_2 = a2*phi_2;
y_3 = b1*phi_3;
y_4 = b2*phi_4;
y_test = [y_1+y_2+y_3+y_4];
% Or cleaner implementation:
y_hat = phi*thetaLS;

% Plot both the original input and output data, and the output from the new
% system model
plot(T,Usys,':',T,Y,':',T,y_test,'LineWidth',2); 
title('System Model Test')
xlabel('time (seconds)')
ylabel('volts')
legend('System Input','System Output','Simulated Output')

% Check the error with residuals
e = Y-y_hat;
plot(T,e,'.');
title('System Model Residuals (Error)');
xlabel('time (seconds)');
ylabel('Error (voltage)');

% ------------------------- STEP 3 ------------------------- %
% Transfer function sys
sys = tf([b1,b2],[1,a1,a2]);
U = M(2:end,2);  % U is the input data
L = length(U);
Fs = 50;    % Hz


% Input frequency domain ID using fft
fft_inp = fft(U);
P2 = abs(fft_inp/L);
P1 = P2(1:L/2+1);  % graph on y-axis
P1(2:end-1) = 2*P1(2:end-1);
freq_inp = Fs*(0:(L/2))/L;  % graph on x-axis

% Plot frequency response
plot(freq_inp,P1);
title('Input Frequency Response');
xlabel('frequency (Hz)')
ylabel('Magnitude')



