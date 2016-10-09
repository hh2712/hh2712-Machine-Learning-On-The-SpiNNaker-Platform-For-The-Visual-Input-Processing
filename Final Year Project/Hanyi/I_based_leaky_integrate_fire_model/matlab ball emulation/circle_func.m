function circle_func(src, evt, stop)
    x_old=1;
    y_old=1;
    x_new=floor(500*rand(1))
    y_new=floor(500*rand(1))
    %% ------------------------------------------------------------%
    THETA=linspace(0,2*pi,1000);
    RHO=ones(1,1000)*100;
    [X_orig,Y_orig] = pol2cart(THETA,RHO);
    X_array=X_orig+x_new;
    Y_array=Y_orig+y_new;
    circle=fill(X_array,Y_array,'w-');
    
    %% -----------------------------------------------------------%

 while(strcmp(stop.Label,'Stop'))
        while(x_new==x_old)
            x_new=floor(500*rand(1))
        end
        while(y_new==y_old)
            y_new=floor(500*rand(1))
        end
         stepx=-1*(x_old-x_new)/10;
         stepy=-1*(y_old-y_new)/10;
         
        x=x_old:stepx:x_new
        y=y_old:stepy:y_new;

            for j=2:10
                  X_array=X_orig+x(j);
                  Y_array=Y_orig+y(j)
                set(circle,'XData',X_array);
                set(circle,'YData',Y_array);
                pause(0.05);
                if(strcmp(stop.Label,'Stop')==0)
                    break
                end
            end
       x_old=x_new;
       y_old=y_new;
    end
  set(circle , 'Xdata', [], 'Ydata', [] );
  stop.Label='Stop';
  return
end