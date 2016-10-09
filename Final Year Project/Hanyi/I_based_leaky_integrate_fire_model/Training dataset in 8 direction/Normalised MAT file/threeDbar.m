clear all;
load trained_weight0;
NeuronID = 1;
v = trained_weight(:,NeuronID);
weight = vec2mat(v,16); 
figure
b = bar3(weight);
axis([0 17 0 17 0 1]);

xlabel('X');
ylabel('Y');
zlabel('Synaptic Weight');
set(gca,'Ydir','Normal')

for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
cb = colorbar();
caxis([0, 1])
cb.Label.String = 'Weight Magnitude';
title('Synaptic Weight 3D Histogram');
