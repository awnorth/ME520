
M = readmatrix('data_chirpIO.csv');

Y = M(1:end-1,1);  % Y = first column of M
phi_1 = -M(2:end,3);
phi_2 = -M(1:end-1,3);
phi_3 = M(2:end,2);
phi_4 = M(1:end-1,2);
phi = [phi_1 phi_2 phi_3 phi_4];
phiT = (phi)';

thetaLS = inv(phiT*phi)*phiT*Y;

a1 = thetaLS(1,1);
a2 = thetaLS(2,1);
b1 = thetaLS(3,1);
b2 = thetaLS(4,1);

sys = tf([b1,b2],[1,a1,a2]);
U = M(2:end,2);
L = length(U);
Fs = 50;    % Hz

fft_inp = fft(U);

P2 = abs(fft_inp/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

freq_inp = Fs*(0:(L/2))/L;

plot(freq_inp,P1);
