close all
clear all
clc
f = figure('MenuBar','None');
menu = uimenu(f,'Label','Start simulation');
stop=uimenu(f,'Label','Stop','Callback',{@stop_func},... 
           'Separator','on','Accelerator','X');
quit=uimenu(f,'Label','Quit','Callback',{@exit_func},... 
           'Separator','on','Accelerator','Q');
%arrows=uimenu(menu,'Label','Arrows', 'Callback',{@arrow_func, stop}); 
box=uimenu(menu,'Label','Horizontal Box', 'Callback',{@box_func, stop}); 
vertical_box=uimenu(menu,'Label','Vertical Box', 'Callback',{@box_vertical, stop}); 
blink_box=uimenu(menu,'Label','blinking box', 'Callback',{@blinking_box, stop}); 
random_blink=uimenu(menu,'Label','Randomly blinking box', 'Callback',{@blinking_random, stop}); 
%random_box=uimenu(menu,'Label','Randomly moving box', 'Callback',{@random_func, stop});
%color_box=uimenu(menu,'Label','Blinking', 'Callback',{@static_box, stop});
circle=uimenu(menu,'Label','Bouncing circle', 'Callback',{@circle_func, stop}); 
cir_in_dir = uimenu(menu,'Label','Moving circle', 'Callback',{@cir_dir_func, stop}); 
%hello=uimenu(menu,'Label','HELLO', 'Callback',{@hello_func, stop}); 
hold all;
 axis([0,1000,0,1000]); 
%axis square;
 set(gcf, 'color', [0 0 0]);
 set(gca, 'color', [0 0 0]);
 set(gca,'xtick',[]);
 set(gca,'xticklabel',[]);
 set(gca,'ytick',[])
 set(gca,'yticklabel',[]);

