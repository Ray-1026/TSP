close all

rng('shuffle');
pso = 0;
sa = 0;
pso_sa = 0;

for k = 1:30
    fprintf('Loop %d\n', k);
    num = randi([30 50]);
    shape = [80, 60];

    numbers = randperm(4800, num);

    gbest = PSO(numbers, shape);
    disp(gbest);
    pso = pso + gbest.cost;

    [SA2.route, SA2.cost] = SA(numbers, shape, 1000);
    disp(SA2);
    pso_sa = pso_sa + SA2.cost;

    [SA1.route, SA1.cost] = SA(numbers, shape, 1000);
    disp(SA1);
    sa = sa + SA1.cost;
end

fprintf('PSO : %d\n', pso);
fprintf('SA : %d\n', sa);
fprintf('PSO + SA : %d\n', pso_sa);


%% function
function dist = Distance(x, y)
    dist = abs(x(1)-y(1))+abs(x(2)-y(2));
end

function [route_best, energy_best] = SA(numbers, shape, t)
    n = length(numbers);
    initial_temp = t;
    temperature = initial_temp;
    res = 1e-3;
    ratio = 0.9;
    markov_len = 500;

    energy_current = inf;
    energy_best = inf;

    route_new = numbers;
    route_current = route_new;
    route_best = route_new;

    while temperature > res
        for i = 1:markov_len
            if rand > 0.5
                a = 0;
                b = 0;
                while (a==b)
                    a = ceil(rand*n);
                    b = ceil(rand*n);
                end
                temp = route_new(a);
                route_new(a) = route_new(b);
                route_new(b) = temp;
            else
                factor = randperm(n, 3);
                factor = sort(factor);
                a = factor(1);
                b = factor(2);
                c = factor(3);
                temp = route_new(a:b);
                route_new(a:a+c-b-1) = route_new(b+1:c);
                route_new(a+c-b:c) = temp;
            end
            
            energy_new = 0;
            for j = 1:n-1
                energy_new = energy_new + Distance([idivide(route_new(j)-1, int32(shape(2))), mod(route_new(j)-1, shape(2))], [idivide(route_new(j+1)-1, int32(shape(2))), mod(route_new(j+1)-1, shape(2))]);
            end
            
            energy_new = energy_new + idivide(route_new(1)-1, int32(shape(2))) + idivide(route_new(n)-1, int32(shape(2))) + 2;

            if energy_new < energy_current
                energy_current = energy_new;
                route_current = route_new;
                
                if energy_new < energy_best
                    energy_best = energy_new;
                    route_best = route_new;
                end
            else
                if rand < exp(double(-(energy_new - energy_current)/temperature))
                    energy_current = energy_new;
                    route_current = route_new;
                else
                    route_new = route_current;
                end
            end
        end

        temperature = temperature * ratio;
    end
end


function fitness = ComputeFitness(numbers, shape, pos)
    fitness = 0;
    for i = 1:length(pos)-1
        x = numbers(int32(pos(i)));
        y = numbers(int32(pos(i+1)));
        fitness = fitness + Distance([idivide(x-1, int32(shape(2))), mod(x-1, shape(2))], [idivide(y-1, int32(shape(2))), mod(y-1, shape(2))]);
    end
    fitness = fitness + idivide(numbers(pos(1)), int32(shape(2))) + idivide(numbers(pos(end)), int32(shape(2))) + 2;
end


function gbest = PSO(numbers, shape)
    n = length(numbers);
    n_iterations = 1000;
    population = 100;

    alpha = 1;
    beta = 0.5;

    particle.route = [];
    particle.pbest = [];
    particle.current_cost = [];
    particle.best_cost = [];
    particle.v = [];

    particles = repmat(particle, population, 1);

    gbest.route = [];
    gbest.cost = inf;

    for i = 1:population
        solution = randperm(n);
        cost = ComputeFitness(numbers, shape, solution);

        particles(i).route = solution;
        particles(i).pbest = solution;
        particles(i).current_cost = cost;
        particles(i).best_cost = cost;

        if cost<gbest.cost
            gbest.route = solution;
            gbest.cost = cost;
        end
    end

    for i = 1:n_iterations
        for m = 1:population
            if particles(m).best_cost < gbest.cost
                gbest.route = particles(m).pbest;
                gbest.cost = particles(m).best_cost;
            end
        end
        for j = 1:population
            temp_v = [];
            solution_gbest = gbest.route;
            solution_pbest = particles(j).pbest;
            solution_particle = particles(j).route;
            
            % pbest-x(t-1)
            for k = 1:n
                if solution_particle(k) ~= solution_pbest(k)
                    swap_op = [k, find(solution_pbest == solution_particle(k)), alpha];
                    temp_v = [temp_v; swap_op];

                    aux = solution_pbest(swap_op(1));
                    solution_pbest(swap_op(1)) = solution_pbest(swap_op(2));
                    solution_pbest(swap_op(2)) = aux;
                end
            end

            % gbest-x(t-1)
            for k = 1:n
                if solution_particle(k) ~= solution_gbest(k)
                    swap_op = [k, find(solution_gbest == solution_particle(k)), beta];
                    temp_v = [temp_v; swap_op];

                    aux = solution_gbest(swap_op(1));
                    solution_gbest(swap_op(1)) = solution_gbest(swap_op(2));
                    solution_gbest(swap_op(2)) = aux;
                end
            end

            particles(j).v = temp_v;

            for so = 1:size(temp_v, 1)
                if rand <= temp_v(so, 3)
                    aux = solution_particle(temp_v(so, 1));
                    solution_particle(temp_v(so, 1)) = solution_particle(temp_v(so, 2));
                    solution_particle(temp_v(so, 2)) = aux;
                end
            end

            particles(j).route = solution_particle;
            particles(j).current_cost = ComputeFitness(numbers, shape, solution_particle);

            if particles(j).current_cost < particles(j).best_cost
                particles(j).pbest = particles(j).route;
                particles(j).best_cost = particles(j).current_cost;
            end
        end
        alpha = alpha * 0.999;
        beta = beta * 1.001;
    end

    for i = 1:n
        gbest.route(i) = numbers(gbest.route(i));
    end
end
