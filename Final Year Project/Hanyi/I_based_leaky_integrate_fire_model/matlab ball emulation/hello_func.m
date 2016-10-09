function hello_func(src, evt, stop)
 
    while(strcmp(stop.Label,'Stop'))
        growH(stop);
        growE(stop);
        growL(stop);
        growL(stop);
        growO(stop);
%        H=patch(h_x, h_y, 'w');
%        blink(H, stop);
%        E=patch(e_x, e_y, 'w');
%        blink(E, stop);
%        L=patch(l_x, l_y, 'w');
%        blink(L, stop);
%        L=patch(l_x, l_y, 'w');
%        blink(L, stop);
%        O=patch(o_x, o_y, 'w');
%        blink(O, stop);
   end
 
   stop.Label='Stop';
   return
end

function growH(stop)
   ratio=20;
  left=(500-(5*ratio))/2;
  bottom=(500-(5*ratio))/2;
  % coordinates for letter H
  h_y=[bottom bottom+5*ratio bottom+5*ratio bottom+3*ratio bottom+3*ratio bottom+5*ratio bottom+5*ratio bottom bottom bottom+2*ratio bottom+2*ratio bottom];
  h_x=[left left left+ratio left+ratio left+2*ratio left+2*ratio left+3*ratio left+3*ratio left+2*ratio left+2*ratio left+ratio left+ratio];
  H=patch(h_x, h_y, 'w');
  
 for rat=20:80
    left=(500-(5*rat))/2;
    bottom=(500-(5*rat))/2;
   
    % coordinates for letter H
    h_y=[bottom bottom+5*rat bottom+5*rat bottom+3*rat bottom+3*rat bottom+5*rat bottom+5*rat bottom bottom bottom+2*rat bottom+2*rat bottom];
    h_x=[left left left+rat left+rat left+2*rat left+2*rat left+3*rat left+3*rat left+2*rat left+2*rat left+rat left+rat];
    set(H, 'XData', h_x);
    set(H, 'YData', h_y);  
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
 end
  pause(1);
  set(H, 'Xdata', [], 'Ydata', [] );
end

function growE(stop)
   ratio=20;
  left=(500-(5*ratio))/2;
  bottom=(500-(5*ratio))/2;
     %coordinates for E
    e_x=[left left left+3*ratio left+3*ratio left+ratio left+ratio left+3*ratio left+3*ratio left+ratio left+ratio left+3*ratio left+3*ratio];
    e_y=[bottom bottom+5*ratio bottom+5*ratio bottom+4*ratio bottom+4*ratio bottom+3*ratio bottom+3*ratio bottom+2*ratio bottom+2*ratio bottom+ratio bottom+ratio bottom];
  E=patch(e_x, e_y, 'w');
  
 for ratio=20:80
  %  left=(500-(5*rat))/2;
    bottom=(500-(5*ratio))/2;
   
   %coordinates for E
    e_x=[left left left+3*ratio left+3*ratio left+ratio left+ratio left+3*ratio left+3*ratio left+ratio left+ratio left+3*ratio left+3*ratio];
    e_y=[bottom bottom+5*ratio bottom+5*ratio bottom+4*ratio bottom+4*ratio bottom+3*ratio bottom+3*ratio bottom+2*ratio bottom+2*ratio bottom+ratio bottom+ratio bottom];
    set(E, 'XData', e_x);  
    set(E, 'YData', e_y);  
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
 end
  pause(1);
  set(E, 'Xdata', [], 'Ydata', [] );
end

function growL(stop)
   ratio=20;
  left=(500-(5*ratio))/2;
  bottom=(500-(5*ratio))/2;
  %coordinates for L
  l_x=[left left left+ratio left+ratio left+3*ratio left+3*ratio];
  l_y=[bottom bottom+5*ratio bottom+5*ratio bottom+ratio bottom+ratio bottom];
  L=patch(l_x, l_y, 'w');
  
 for ratio=20:80
    left=(500-(5*ratio))/2;
    bottom=(500-(5*ratio))/2;
   
      %coordinates for L
    l_x=[left left left+ratio left+ratio left+3*ratio left+3*ratio];
    l_y=[bottom bottom+5*ratio bottom+5*ratio bottom+ratio bottom+ratio bottom];
    set(L, 'XData', l_x);
    set(L, 'YData', l_y);  
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
 end
  pause(1);
  set(L, 'Xdata', [], 'Ydata', [] );
end

function growO(stop)
  ratio=20;
  left=(500-(5*ratio))/2;
  bottom=(500-(5*ratio))/2;
  N=1000;
  THETA=linspace(0,2*pi,N);
    r=2.5*ratio;
    RHO=ones(1,N)*r;
   [outx,outy] = pol2cart(THETA,RHO);
    
    RHO2=ones(1,N)*(ratio);
    [inx,iny] = pol2cart(THETA,RHO2);
    outx=left+r+outx;
    inx=left+r+inx;
    outy=bottom+r+outy;
    iny=bottom+r+iny;
    %coordinates for O
    o_x=[outx inx];
    o_y=[outy iny];
  
  O=patch(o_x, o_y, 'w');
  
 for ratio=20:80
    left=(500-(5*ratio))/2;
    bottom=(500-(5*ratio))/2;
   
    r=2.5*ratio;
    RHO=ones(1,N)*r;
   [outx,outy] = pol2cart(THETA,RHO);
    
    RHO2=ones(1,N)*(ratio);
    [inx,iny] = pol2cart(THETA,RHO2);
    outx=left+r+outx;
    inx=left+r+inx;
    outy=bottom+r+outy;
    iny=bottom+r+iny;
    %coordinates for O
    o_x=[outx inx];
    o_y=[outy iny];
    set(O, 'XData', o_x);  
    set(O, 'YData', o_y);  
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
 end
  pause(1);
  set(O, 'Xdata', [], 'Ydata', [] );
  return;
end

function blink(letter, stop)
   rgb=0:0.1:1;
  for i=1:10
    set(letter, 'FaceColor', [rgb(i) rgb(i) rgb(i)]);
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
   end
   pause(1);
   for i=1:10
    set(letter, 'FaceColor', [1-rgb(i) 1-rgb(i) 1-rgb(i)]);
    pause(0.1);
    if(strcmp(stop.Label,'Stop')==0)
        break
    end
   end
    set(letter , 'Xdata', [], 'Ydata', [] );
   return
end