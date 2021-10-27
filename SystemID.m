% ME520 System ID Assignment - 10_27_2021
% Taylor Ayars
% Andrew North

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
G_hat = y_hat./Usys';

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
hold on;

% Compare to step 2
u = Usys';
fft_u = fft(u);
fft_y_hat = fft(y_hat);
fft_G_model = fft_y_hat./fft_u;
P2gm = abs(fft_G_model/L);
P1gm = P2gm(1:L/2+1);  % graph on y-axis
P1gm(2:end-1) = 2*P1gm(2:end-1);
freq_G_model = Fs*(0:(L/2))/L;  % graph on x-axis

plot(freq_G_model,P1gm);
legend('Measured Freq Response','Model Freq Response');

% ------------------------- STEP 4 ------------------------- %
U = M(2:end,2);
Y = M(2:end,3);  % Updated Y length
deltaT = 0.02;  % Approximate time step in data
% System ID Toolbox. 

% ----------------- Weighted Least Squares ----------------- %
MM = readmatrix('data_SystemID_ChangeDynamics.csv');
N = 289;  % Number of data measurements

% Parameters
aw1 = [];
aw2 = [];
bw1 = [];
bw2 = [];

for N = 1:285
    YY = MM(1:N,3);
    UU = MM(1:N,2);
    TT = MM(1:N,1)'; % time matrix
    phi_1 = -MM(2:N+1,3);  % -y(k-1)  output
    phi_2 = -MM(3:N+2,3);    % -y(k-2)  output
    phi_3 = MM(2:N+1,2);   % u(k-1)   input
    phi_4 = MM(3:N+2,2);     % u(k-2)   input
    phi = [phi_1 phi_2 phi_3 phi_4];
    
    gamma = .99; % 0.2
    
    WW = eye(N);
    % Create Weight matrix
    for row = 1:N
        for col = 1:N;
            if row == col
                WW(row, col) = gamma^(N-row)-gamma^(N-row+1);
            end
        end
    end
    
    thetaWLS = inv(phi'*WW*phi)*phi'*WW*YY;
    y_hat_Weighted = phi*thetaWLS;
    
    % plot(TT,UU,':',TT,YY,':',TT,y_hat_Weighted,'LineWidth',2); 
    % plot(TT,y_hat_Weighted,'LineWidth',2); 
    % calculate the new parameters
    aw1(N) = thetaWLS(1,1);
    aw2(N) = thetaWLS(2,1);
    bw1(N) = thetaWLS(3,1);
    bw2(N) = thetaWLS(4,1);
end

figure();
subplot(2,1,1);
plot(TT,aw1);
title('A1 Parameter')
xlabel('time (seconds)')
ylabel('A1 Value')
subplot(2,1,2);
plot(TT,aw2);
title('A2 Parameter')
xlabel('time (seconds)')
ylabel('A2 Value')
