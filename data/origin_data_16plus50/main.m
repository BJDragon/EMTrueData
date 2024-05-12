clear;clc;close all
data_30= readmatrix("30_all.xlsx");data_30=data_30(2:end,:);
data_90= readmatrix("90_all.xlsx");data_90=data_90(2:end,:);
freq = readmatrix("freq.xlsx");

%%
for i = 1:66
    y1 = data_30(:, i);
    % 自定义参数查找峰值
    [pks1, locs1] = findpeaks(-y1, ...
        'SortStr','descend', ...
        'MinPeakHeight', 10, ...
        'MinPeakProminence',3);
    
    y2 = data_90(:, i);
    % 自定义参数查找峰值
    [pks2, locs2] = findpeaks(-y2, ...
        'SortStr','descend', ...
        'MinPeakHeight', 10, ...
        'MinPeakProminence',3);

    % 绘制信号和自定义参数查找的峰值
    figure(i);
    plot(freq, y1, freq(locs1), -pks1, '*');
    hold on
    plot(freq, y2, freq(locs2), -pks2, 'ro');
    title('信号和自定义参数查找的峰值');
    xlabel('频率');
    ylabel('dB');
    disp(['No.',num2str(i),' min_value'])
    min_value=ones(length(locs2),length(locs1));
    for j =1:length(locs2)
        for ij =1:length(locs1)
            min_value(j,ij)=freq(locs2(j))-freq(locs1(ij));
        end
        [~, idx]=min(abs(min_value(j,:)));
        disp(min_value(j,idx))
    end
    clc;close
end
