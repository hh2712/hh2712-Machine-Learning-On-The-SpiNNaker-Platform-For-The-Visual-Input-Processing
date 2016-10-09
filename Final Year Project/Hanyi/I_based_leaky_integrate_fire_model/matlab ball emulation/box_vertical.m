function box_vertical(src, evt, stop)
%     vidObj = VideoWriter('vertical.avi');
%     open(vidObj);
   % box  = plot([0 0],[100, 0],'w-','LineWidth',200)
   box=patch([0 0 100 100], [0 100 100 0], 'w');
   while(strcmp(stop.Label,'Stop'))
         for x=1:100:500
            for y=1:10:500
               set(box,'XData',[x x x+100 x+100]);
                set(box,'YData',[y y+100 y+100 y ]);
                pause(0.1);
%                 currFrame = getframe;
%                 writeVideo(vidObj,currFrame);
            
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
%      close(vidObj);
     return
end