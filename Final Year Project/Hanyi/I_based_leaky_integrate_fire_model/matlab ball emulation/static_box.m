function static_box(src, evt, stop)
    box  = plot([250 250],[200, 300],'w-','LineWidth',200)
     rgb=0:0.1:1;
   while(strcmp(stop.Label,'Stop'))
       for i=1:10
        set(box, 'color', [rgb(i) rgb(i) rgb(i)]);
        pause(0.15);
        if(strcmp(stop.Label,'Stop')==0)
            break
        end
       end
       pause(1);
       for i=1:10
        set(box, 'color', [1-rgb(i) 1-rgb(i) 1-rgb(i)]);
        pause(0.15);
        if(strcmp(stop.Label,'Stop')==0)
            break
        end
       end
   end
   
   set(box , 'Xdata', [], 'Ydata', [] );
   stop.Label='Stop';
   return
end