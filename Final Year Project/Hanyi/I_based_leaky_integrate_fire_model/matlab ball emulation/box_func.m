function box_func(src, evt, stop)
   % box  = plot([0 0],[100, 0],'w-','LineWidth',200)
   box=patch([0 0 100 100], [0 100 100 0], 'w');
   while(strcmp(stop.Label,'Stop'))
        for y=1:50:500
            for x=1:20:500
               set(box,'XData',[x x x+100 x+100]);
                set(box,'YData',[y y+100 y+100 y ]);
                pause(0.1);
                if(strcmp(stop.Label,'Stop')==0)
                    break
                end
            end
            if(strcmp(stop.Label,'Stop')==0)
                break
           end
        end
   end
     set(box , 'Xdata', [], 'Ydata', [] );
     stop.Label='Stop';
     return
end