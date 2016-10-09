function blinking_box(src, evt, stop)
   %box  = plot([0 0],[100, 0],'w-','LineWidth',100)
   box=patch([0 0 100 100], [0 100 100 0], [1, 1, 1]);
   rgb=0:0.1:1;
   while(strcmp(stop.Label,'Stop'))
         for x=1:150:500
            for y=1:125:500
                     for j=1:10
                        set(box, 'FaceColor', [rgb(j) rgb(j) rgb(j)]);
                        pause(0.1);
                        
                     end
                      for i=1:10
                        set(box, 'FaceColor', [1-rgb(j) 1-rgb(j) 1-rgb(j)]);
                        pause(0.1);
                      end  
                set(box,'XData',[x x x+100 x+100]);
               set(box,'YData',[y y+100 y+100 y ]);
            if(strcmp(stop.Label,'Stop')==0)
                break
            end
           end
        end
   end
     set(box , 'Xdata', [], 'Ydata', [] );
     stop.Label='Stop';
     return
end