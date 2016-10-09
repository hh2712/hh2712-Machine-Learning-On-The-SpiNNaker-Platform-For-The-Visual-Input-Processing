function random_func(src, evt, stop)
    x_old=1;
    y_old=1;
    x_new=floor(500*rand(1))
    y_new=floor(500*rand(1))
    %box  = plot([x_old y_old],[100, 0],'w-','LineWidth',200)
    box=patch([x_old x_old x_old+100 x_old+100], [y_old y_old+100 y_old+100 y_old], 'w');
    while(strcmp(stop.Label,'Stop'))
        while(x_new==x_old)
            x_new=floor(500*rand(1))
        end
        while(y_new==y_old)
            y_new=floor(500*rand(1))
        end
        stepx=-1*(x_old-x_new)/10
        stepy=-1*(y_old-y_new)/10
        x=x_old:stepx:x_new
        y=y_old:stepy:y_new

            for j=2:10
                set(box,'XData',[x(j) x(j) x(j)+100 x(j)+100]);
                set(box,'YData',[y(j) y(j)+100 y(j)+100 y(j) ]);
                pause(0.1);
                if(strcmp(stop.Label,'Stop')==0)
                 break
                end
            end
       x_old=x_new;
       y_old=y_new;
    end
   set(box , 'Xdata', [], 'Ydata', [] );
   stop.Label='Stop';
   return
end