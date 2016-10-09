x = 0:0.001:0.05;
y = exp(-100*x)-exp(-200*x);
z = 20*0.195*y;
plot(x,z)
xlabel('time')
ylabel('membrane potential')
title('Matlab Simulated Membrane Potential')