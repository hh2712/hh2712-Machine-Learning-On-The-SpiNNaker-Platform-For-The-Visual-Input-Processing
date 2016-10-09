function cir_dir_func(src, evt, stop)
    x_old=0;
    y_old=0;
    x_new=1000;
    y_new=1000;
    prev_angle = 0;
    angle = 270;%change this value to change the angle of ball movement
    offsety= 500;
    offsetx = 500;
    %% ------------------------------------------------------------%
    THETA=linspace(0,2*pi,1000);
    RHO=ones(1,1000)*140;
    [X_orig,Y_orig] = pol2cart(THETA,RHO);
    X_array=X_orig+x_new;
    Y_array=Y_orig+y_new;
    circle=fill(X_array,Y_array,'w-');
    
    %% -----------------------------------------------------------%
 txt = uicontrol('Style','text','Position',[20 20 120 20],'String','Moving Angle');  
%txt2 = uicontrol('Style','text','Position',[250 20 300 20],'String',str(angle));  
%gamma = uicontrol('Style', 'slider','Min',0,'Value',0 ,'Max',360, 'Position', [150 20 120 20],'Callback',@ball_angle);
 while(strcmp(stop.Label,'Stop'))
        
        %angle = getGlobalx;
%         if(isempty(angle))
%             angle = 0;
%         end
        
        if(angle~=prev_angle)
            x_old=offsetx-offsety*cos(angle*pi/180);
            y_old=offsety-offsetx*sin(angle*pi/180);
            x_new=offsetx+offsety*cos(angle*pi/180);
            y_new=offsety+offsetx*sin(angle*pi/180);
            prev_angle = angle;
        else
            x_old=offsetx-offsety*cos(angle*pi/180);
            y_old=offsety-offsetx*sin(angle*pi/180);
            x_new=offsetx+offsety*cos(angle*pi/180);
            y_new=offsety+offsetx*sin(angle*pi/180);   
        end
        stepx=-1*(x_old-x_new)/125;
        stepy=-1*(y_old-y_new)/125;
        if(stepx==0)
            x = offsetx*ones(1,126)
        else
            x = x_old:stepx:x_new
        end
        
        if(stepy==0)
             y = offsety*ones(1,126)
        else
             y=y_old:stepy:y_new;
        end

        for j=2:100
            X_array=X_orig+x(j);
            Y_array=Y_orig+y(j);
            set(circle,'XData',X_array);
            set(circle,'YData',Y_array);
            pause(0.1);
            if(strcmp(stop.Label,'Stop')==0)
                break
            end
        end
    end
  set(circle , 'Xdata', [], 'Ydata', [] );
  stop.Label='Stop';
  return
end
  
  function ball_angle(source, callbackdata)
      setGlobalx(source.Value);
  end
  
  
  function setGlobalx(angle)
    global x
    x = angle;
  end
  
  function r = getGlobalx
    global x
    r = x;
  end