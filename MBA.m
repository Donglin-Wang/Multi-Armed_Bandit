clear
clc
close all

% options = [0.3, 0.8, 0.8, 0.5];
% rewards = [0, 1, 3, 1];
% costs = [8, 0, 0, 4];
options = [0.8, 0.5, 0.7, 0.4, 0.3, 0.05, 0.1];
rewards = [3, 6, 7, 1, 2, 1, 3];
costs = [0,0,0,0,0,0,0];

K = 5000

[select_plt, reward_plt] = ep_greedy(0.1, options, rewards, costs, K);
[select_plt2, reward_plt2] = UCB(options, rewards, costs, K);
figure
hold on
title('Arm Selection Ratio - Ep Greedy');
for i = 1:7
    plot(select_plt(:,i));
end
hold off

figure
hold on
title('Arm Selection Ratio - UCB');
for i = 1:7
    plot(select_plt2(:,i));
end
hold off

expectation = 4.9 * ones(1, K);



figure
title('UCB Average Regret vs Expected Average Regre')
hold on
plot(reward_plt2);
plot(reward_plt);
plot(expectation);
legend('UCB', 'EP Greedy', 'Optimal');
hold off



function plot_policy(options, rewards, costs, reward_plt, select_plt, exp)
    iterations = size(select_plt, 1);
    option_num = size(options, 2);
    expectation = exp .* ones(iterations);
    figure
    hold on
    for i = 1:option_num
        plot(select_plt(:,i));
    end
    hold off
    figure
    hold on
    plot(expectation);
    plot(reward_plt);
    hold off
    
end

function [select_plt, reward_plt] = ep_greedy(epsi, options, rewards, costs, iterations)
    option_num = size(options, 2);
    total = 0;
    avg_reward = zeros(1, option_num);
    select_cnt = zeros(1, option_num);
    reward_plt = zeros(1, iterations);
    select_plt = zeros(iterations, option_num);
    for i = 1:iterations
        if epsi < rand
            target = randi(option_num);
        else
            [~, target] = max(avg_reward);
        end
        reward = rand_pull(options, rewards, costs, target);
        total = total + reward;
        reward_plt(i) = total / i;
        select_cnt(target) = select_cnt(target) + 1;
        avg_reward(target) = ( (select_cnt(target) - 1) * avg_reward(target) + reward) / select_cnt(target);
        select_plt(i, :) = select_cnt ./ i;
    end
end





function reward = rand_pull(options, rewards, costs, arm)
    if(options(arm) > rand)
        reward = rewards(arm) - costs(arm);
    else
        reward = -costs(arm);
    end

end



function [select_plt,reward_plt] = UCB(options, rewards, costs, trial_num)
    target = 0;
    option_num = size(options,2);
    selected = zeros(1, option_num);
    reward_cnt = zeros(1, option_num);
    select_plt = zeros(trial_num, option_num);
    reward_plt = zeros(1, trial_num);
    total = 0;
    for i = 1:trial_num
        max_thresh = 0;
        for j = 1:option_num
            if selected(j) > 0
                average = reward_cnt(j) / selected(j);
                buf = sqrt( 2 * log(i) / selected(j) );
                thresh = average + buf;
            else
                thresh = 1e800;
            end
            if thresh > max_thresh
                max_thresh = thresh;
                target = j;
            end
        end
        selected(target) = selected(target) + 1;
        reward = rand_pull(options, rewards, costs, target);
        total = total + reward;
        reward_plt(i) = total / i;
        reward_cnt(target) = reward_cnt(target) + reward;
        select_plt(i, :) = selected ./ i;
    end
end


