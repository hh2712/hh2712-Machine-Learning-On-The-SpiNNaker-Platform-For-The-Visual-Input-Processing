function arrow_func(src,evt, stop)
    disp('blablabla');
    %right arrow
    x_left=0;
    x_middle=180;
    x_right=300;

    y_top=500;
    y_bottom=000;
    y_diff=180;

    Y=[y_top-y_diff y_top-y_diff y_top (y_top+y_bottom)/2 y_bottom y_bottom+y_diff y_bottom+y_diff];
    X=[x_left x_middle x_middle x_right x_middle x_middle x_left];
    arrow=patch(X,Y,'w-');

   % back = uicontrol('Style', 'togglebutton', 'String','Back', 'Position', [0, 400, 50, 50]);
   
   while(strcmp(stop.Label,'Stop'))
        for i=1:10:510
            x_left=i;
            x_middle=x_left+180;
            x_right=x_left+400;
            X=[x_left x_middle x_middle x_right x_middle x_middle x_left];
            set(arrow,'XData', X);
            pause(0.05);

            if(strcmp(stop.Label,'Stop')==0)
                break
            end
        end
   end
   set(arrow , 'Xdata', [], 'Ydata', [] );
   stop.Label='Stop';
   return
end
%left arrow
% Y=[230 230 300 150 0 70 70]
% X=[0 100 100 150 100 100 0]
% patch(X,Y,'w-')