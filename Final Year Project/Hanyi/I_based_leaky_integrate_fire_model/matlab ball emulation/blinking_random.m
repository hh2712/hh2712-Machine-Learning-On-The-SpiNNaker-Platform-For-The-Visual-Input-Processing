function blinking_random(src, evt, stop)
   %box  = plot([0 0],[100, 0],'w-','LineWidth',100)
%    vidObj = VideoWriter('blinking_random.avi');
%    open(vidObj);
   box=patch([0 0 100 100], [0 100 100 0], [1, 1, 1]);
   rgb=0:0.1:1;
   x_old=0;
   x_new=0;
   y_new=0;
   y_old=0;
   while(strcmp(stop.Label,'Stop'))
        while(x_new==x_old)
            x_new=floor(450*rand(1));
        end
        while(y_new==y_old)
            y_new=floor(450*rand(1));
        end
        for j=1:10
            set(box, 'FaceColor', [rgb(j) rgb(j) rgb(j)]);
%             currFrame = getframe;
%             writeVideo(vidObj,currFrame);
%             currFrame = getframe;
%             writeVideo(vidObj,currFrame);
            pause(0.1);
        end
        for i=1:10
            set(box, 'FaceColor', [1-rgb(j) 1-rgb(j) 1-rgb(j)]);
           % currFrame = getframe;
           % writeVideo(vidObj,currFrame);
           % currFrame = getframe;
           % writeVideo(vidObj,currFrame);
            pause(0.1);
        end  
       set(box,'XData',[x_new x_new x_new+100 x_new+100]);
       set(box,'YData',[y_new y_new+100 y_new+100 y_new ]);
       x_old=x_new;
       y_old=y_new;
        if(strcmp(stop.Label,'Stop')==0)
            break
        end
   end
  
     set(box , 'Xdata', [], 'Ydata', [] );
     stop.Label='Stop';
%      close(vidObj);
     return
end