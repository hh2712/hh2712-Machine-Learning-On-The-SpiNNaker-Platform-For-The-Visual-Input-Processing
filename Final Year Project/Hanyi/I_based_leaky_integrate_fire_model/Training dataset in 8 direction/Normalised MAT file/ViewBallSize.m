clear all;
load('new45d_2.mat');

scatter3(ts,X,Y,'filled');
axis([0 200 0 16 0 16]);
title('3D Scattering Plot of Neuron Population (Right to Left)')
xlabel('Time/ms');
ylabel('X');
zlabel('Y');
set(gca,'YDir','Reverse')