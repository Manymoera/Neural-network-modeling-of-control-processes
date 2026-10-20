%task1()

function task1()
    omega = 1.32;
    t_start = 0;
    t_end = 20;
    N = 10000;
    
    t_eval = linspace(t_start, t_end, N);
    
    data = step_a_generate_data(t_eval, omega);
    
    derivatives = step_b_dx_calc(data, omega);
    
    step_c_parameter_estimation(data, derivatives, omega);
    
    step_d_noisy_estimation(data, derivatives, omega)
    
    step_e_finite_difference_estimation(data, omega)
    
    function f = step_a_generate_data(t_eval, omega)
        x0 = [0.0; 0.0];
    
        [t, x] = ode45(@(t, x) odes(t, x, omega), t_eval, x0);
    
        dataset = [t, x];
        
        disp('Step a):')
        disp('   t_k     |    x1(t_k)    |    x2(t_k)');
        disp(dataset(1:5, :));
    
        f = [t, x];
    end
    
    function f = step_b_dx_calc(data, omega)
        t = data(:, 1);
        x1 = data(:, 2);
        x2 = data(:, 3);
    
        dx1 = x2;
        dx2 = -(omega^2) * x1 + sin(t);
        
        dataset = [t, dx1, dx2];
        
        disp('Step b):')
        disp('   t_k     |    dx1(t_k)    |    dx2(t_k)');
        disp(dataset(1:5, :));
    
        f = [dx1, dx2];
    end
    
    function step_c_parameter_estimation(data, derivatives, omega)
        t = data(:, 1);
        x1 = data(:, 2);
    
        dx2_exact = derivatives(:, 2);
    
        a_hat = 0.0;
        P = 0.0001;
        
        N_points = length(t);
        omega_hat_history = zeros(N_points, 1);
    
        for k = 1:N_points
            x1_k = x1(k);
            
            F = -x1_k; 
            
            P = P + (F' * F');
            
            model_val = -a_hat * x1_k + sin(t(k));
            residual = dx2_exact(k) - model_val;
            
            % a_N = a_(N-1) + P'_N * F' * (x_(N-1) - f(x_N, u_N, a_(N-1))
            % f(x_N, u_N, a_(N-1) = -a_(N-1) * x1(t_k) + sint(t_k)
            a_hat = a_hat + (F' / P) * residual;
    
            omega_hat_history(k) = sqrt(max(0, a_hat));
        end
    
        disp('Step с):')
        fprintf('Истинная omega: %.4f\n', omega);
        fprintf('Оценка omega: %.4f\n', omega_hat_history(end));
        disp(' ');
        
        figure;
        plot(t, omega_hat_history, 'LineWidth', 1.5);
        hold on;
        yline(omega, '--r', 'LineWidth', 1.5);
        grid on;
        xlabel('Время t');
        ylabel('Оценка omega');
        legend('Оценка omega', 'Истинное значение omega');
        title('Идентификация параметра omega рекуррентным методом без шума');
    end
    
    function step_d_noisy_estimation(data, derivatives, omega)
        t  = data(:, 1);
        x1 = data(:, 2);
        N_points = length(t);
    
        sigmas = [0.01, 0.1, 1.0];
        rng(42);
    
        disp('Step г):')
        figure;
        for s = 1:length(sigmas)
            sigma = sigmas(s);
    
            dx2_noisy = derivatives(:, 2) + sigma * randn(N_points, 1);
    
            a_hat = 0.0;
            R = 1e-4;
            omega_hat_history = zeros(N_points, 1);
    
            for k = 1:N_points
                F = -x1(k);
                y = dx2_noisy(k) - sin(t(k));
                R = R + F^2;
                a_hat = a_hat + (F / R) * (y - F * a_hat);
                omega_hat_history(k) = sqrt(max(0, a_hat));
            end
    
            fprintf('sigma = %.2f: истинная omega = %.4f, оценка omega = %.4f\n', ...
                sigma, omega, omega_hat_history(end));
    
            subplot(numel(sigmas), 1, s);
            plot(1:N_points, omega_hat_history, 'LineWidth', 1.2);
            hold on;
            yline(omega, '--r', 'LineWidth', 1.5);
            grid on;
            ylim([0, 3]);
            xlabel('Количество использованных данных k');
            ylabel('Оценка omega');
            title(sprintf('Рекуррентный МНК, шум в производных: \\sigma = %.2f', sigma));
            legend('Оценка omega', 'Истинное значение omega', 'Location', 'best');
        end
        disp(' ');
    end
    
    function step_e_finite_difference_estimation(data, omega)
        t = data(:, 1);
        X = data(:, 2:3);
        N_points = length(t);
        h = t(2) - t(1);
    
        dx_fd = zeros(N_points, 2);
        dx_fd(1:end-1, :) = (X(2:end, :) - X(1:end-1, :)) / h;
        dx_fd(end, :) = (X(end, :) - X(end-1, :)) / h;
    
        disp('Step д):')
        disp('   t_k     |   dx1_fd(t_k)   |   dx2_fd(t_k)');
        disp([t(1:5), dx_fd(1:5, :)]);
        fprintf('h = %.6f\n', h);
    
        omega_hat_clean = recursive_lsq_omega(t, X(:, 1), dx_fd(:, 2));
        fprintf('Без шума: истинная omega = %.4f, оценка omega = %.4f, ошибка = %.4f\n', ...
            omega, omega_hat_clean(end), abs(omega - omega_hat_clean(end)));
    
        sigmas = [0.01, 0.1, 1.0];
        rng(42);
    
        figure;
        for s = 1:numel(sigmas)
            sigma = sigmas(s);
    
            dx2_noisy = dx_fd(:, 2) + sigma * randn(N_points, 1);
    
            omega_hat_noisy = recursive_lsq_omega(t, X(:, 1), dx2_noisy);
    
            fprintf('sigma = %.2f: истинная omega = %.4f, оценка omega = %.4f, ошибка = %.4f\n', ...
                sigma, omega, omega_hat_noisy(end), abs(omega - omega_hat_noisy(end)));
    
            subplot(numel(sigmas), 1, s);
            hold on;
            grid on;
            h_noisy = plot(1:N_points, omega_hat_noisy, 'LineWidth', 1.2);
            h_clean = plot(1:N_points, omega_hat_clean, 'g', 'LineWidth', 1.5);
            h_true  = yline(omega, '--r', 'LineWidth', 1.5);
    
            ylim([0, 3]);
            xlabel('Количество использованных данных k');
            ylabel('Оценка \omega');
            title(sprintf('Рекуррентный МНК по конечно-разностным производным: \\sigma = %.2f', sigma));
            legend([h_noisy, h_clean, h_true], ...
                {sprintf('С шумом (\\sigma = %.2f)', sigma), 'Без шума', 'Истинное значение \omega'}, ...
                'Location', 'best');
            hold off;
        end
        disp(' ');
    end
    
    function omega_hat_history = recursive_lsq_omega(t, x1, dx2)
        a_hat = 0.0;
        P = 1e-4;
        N_points = length(t);
        omega_hat_history = zeros(N_points, 1);
    
        for k = 1:N_points
            x1_k = x1(k);
            F = -x1_k;
            P = P + F' * F;
            model_val = -a_hat * x1_k + sin(t(k));
            residual = dx2(k) - model_val;
            a_hat = a_hat + (F' / P) * residual;
            omega_hat_history(k) = sqrt(max(0, a_hat));
        end
    end
    
    function dxdt = odes(t, x, omega)
        x1 = x(1);
        x2 = x(2);
        
        dx1_dt = x2;
        dx2_dt = -(omega^2) * x1 + sin(t);
        
        dxdt = [dx1_dt; dx2_dt];
    end
end

task2(1)

function task2(func_id)

    samples = generate_samples(true);
    step_b_polynomial_fit(samples);
    step_c_trigonometric_fit(samples);

    function sample = generate_samples(use_noise)
        N = 10000;
        sigma = 0.1;
        x0_train = 0.5;
        x0_test = -0.5;

        [f, name, idx] = get_function();

        if use_noise
            [x_train, x_test] = generate_samples_with_noise(f, N, sigma, x0_train, x0_test, idx);
        else
            [x_train, x_test] = generate_samples_without_noise(f, N, x0_train, x0_test);
        end
    
        sample = struct('name', name, 'f', f, 'x_train', x_train, 'x_test', x_test);
    
        disp('Step a):')
        fprintf('%s\n', name);
        fprintf('   обучающая: %d точек, x in [%.3f, %.3f], среднее = %.3f, СКО = %.3f\n', ...
            numel(x_train), min(x_train), max(x_train), mean(x_train), std(x_train));
        fprintf('   тестовая:  %d точек, x in [%.3f, %.3f], среднее = %.3f, СКО = %.3f\n', ...
            numel(x_test), min(x_test), max(x_test), mean(x_test), std(x_test));
        disp(' ');
    
        figure;
    
        subplot(1, 2, 1);
        plot(0:200, x_train(1:201), '.-');
        grid on;
        xlabel('k');
        ylabel('x_k');
        title('Обучающая выборка: первые 200 точек');
    
        subplot(1, 2, 2);
        hold on;
        xx = linspace(min([x_train; x_test]), max([x_train; x_test]), 3000);
        h1 = plot(x_train(1:end-1), x_train(2:end), '.', 'MarkerSize', 3);
        h2 = plot(x_test(1:end-1),  x_test(2:end),  '.', 'MarkerSize', 3);
        h3 = plot(xx, f(xx), 'r.', 'MarkerSize', 3);
        grid on;
        xlabel('x_k');
        ylabel('x_{k+1}');
        title('Пары (x_k, x_{k+1}) и истинная f(x)');
        legend([h1, h2, h3], {'Обучающая', 'Тестовая', 'f(x)'}, 'Location', 'best');
        hold off;
    
        sgtitle(name);

        function [f, name, idx] = get_function()
            funcs = { ...
                @(x) x .* (1 - x.^2), ...                          % (а)
                @(x) cos(x) + sin(2*x), ...                        % (б)
                @(x) (1 - x.^2) .* cos(x) + x .* sin(2*x), ...     % (в)
                @(x) abs(x) - x.^2, ...                            % (г)
                @(x) x + sin(abs(x)), ...                          % (д)
                @(x) sign(sin(x)) ...                              % (е)
            };
        
            names = { ...
                '(а) f(x) = x(1-x^2)', ...
                '(б) f(x) = cos x + sin 2x', ...
                '(в) f(x) = (1-x^2)cos x + x sin 2x', ...
                '(г) f(x) = |x| - x^2', ...
                '(д) f(x) = x + sin|x|', ...
                '(е) f(x) = sign sin x' ...
            };
        
            idx = func_id;
            f = funcs{idx};
            name = names{idx};
        end
    end

    function step_b_polynomial_fit(samples)
        ms = 1:20;                         % степени полинома
        f  = samples.f;

        % Пары (x_k, x_{k+1}) для обучающей и тестовой выборок
        x_tr = samples.x_train(1:end-1);
        y_tr = samples.x_train(2:end);
        x_te = samples.x_test(1:end-1);
        y_te = samples.x_test(2:end);

        rmse_train = zeros(size(ms));
        rmse_test  = zeros(size(ms));
        r2_train   = zeros(size(ms));
        r2_test    = zeros(size(ms));
        cond_A     = zeros(size(ms));
        thetas     = cell(size(ms));

        disp('Step b):')
        fprintf('%s\n', samples.name);
        disp('   m  |  RMSE (обуч.)  |  RMSE (тест)  |  R^2 (обуч.)  |  R^2 (тест)  |  cond(A)');

        for i = 1:numel(ms)
            m = ms(i);

            % Коэффициенты находятся только по обучающей выборке
            [theta, cond_A(i)] = fit_polynomial(x_tr, y_tr, m);
            thetas{i} = theta;

            % polyval ждёт коэффициенты от старшей степени к младшей
            y_hat_tr = polyval(flipud(theta), x_tr);
            y_hat_te = polyval(flipud(theta), x_te);

            rmse_train(i) = calc_rmse(y_tr, y_hat_tr);
            rmse_test(i)  = calc_rmse(y_te, y_hat_te);
            r2_train(i)   = calc_r2(y_tr, y_hat_tr);
            r2_test(i)    = calc_r2(y_te, y_hat_te);

            fprintf('  %2d  |    %.5f    |    %.5f   |    %.5f   |    %.5f  |  %.3e\n', ...
                m, rmse_train(i), rmse_test(i), r2_train(i), r2_test(i), cond_A(i));
        end

        disp(' ');
        disp('Коэффициенты полинома:');
        for i = 1:numel(ms)
            fprintf('m = %d\n', ms(i));
            fprintf('   θ%d = %.5f\n', [0:ms(i); thetas{i}']);
        end
        disp(' ');

        % Эталон: истинная f на тестовой выборке (RMSE около sigma, лучше уже нельзя)
        rmse_true = calc_rmse(y_te, f(x_te));
        r2_true   = calc_r2(y_te, f(x_te));
        fprintf('Истинная f на тестовой выборке: RMSE = %.5f, R^2 = %.5f\n', rmse_true, r2_true);

        figure;

        subplot(1, 2, 1);
        hold on;
        grid on;
        h_tr = plot(ms, rmse_train, 'o-', 'LineWidth', 1.3);
        h_te = plot(ms, rmse_test,  's-', 'LineWidth', 1.3);
        h_nz = yline(rmse_true, '--k', 'LineWidth', 1.3);
        xlabel('Степень полинома m');
        ylabel('RMSE');
        title('RMSE');
        legend([h_tr, h_te, h_nz], {'Обучающая', 'Тестовая', 'Истинная f'}, 'Location', 'best');
        hold off;

        subplot(1, 2, 2);
        hold on;
        grid on;
        h_tr = plot(ms, r2_train, 'o-', 'LineWidth', 1.3);
        h_te = plot(ms, r2_test,  's-', 'LineWidth', 1.3);
        h_nz = yline(r2_true, '--k', 'LineWidth', 1.3);
        xlabel('Степень полинома m');
        ylabel('R^2');
        title('Коэффициент детерминации R^2');
        legend([h_tr, h_te, h_nz], {'Обучающая', 'Тестовая', 'Истинная f'}, 'Location', 'best');
        hold off;

        sgtitle(sprintf('Полиномиальный базис: %s', samples.name));
        disp(' ');
    end

        function step_c_trigonometric_fit(samples)
        ms = 1:10;
        f  = samples.f;

        x_tr = samples.x_train(1:end-1);
        y_tr = samples.x_train(2:end);
        x_te = samples.x_test(1:end-1);
        y_te = samples.x_test(2:end);

        % Основная частота базиса: период вдвое больше диапазона обучающих данных
        omega = pi / (max(x_tr) - min(x_tr));

        rmse_train = zeros(size(ms));
        rmse_test  = zeros(size(ms));
        r2_train   = zeros(size(ms));
        r2_test    = zeros(size(ms));
        cond_P     = zeros(size(ms));
        thetas     = cell(size(ms));

        disp('Step c):')
        fprintf('%s\n', samples.name);
        fprintf('omega = %.5f\n', omega);
        disp('   m  |  RMSE (обуч.)  |  RMSE (тест)  |  R^2 (обуч.)  |  R^2 (тест)  |  cond(P)');

        for i = 1:numel(ms)
            m = ms(i);

            [theta, cond_P(i)] = fit_trigonometric(x_tr, y_tr, m, omega);
            thetas{i} = theta;

            y_hat_tr = trig_basis(x_tr, m, omega) * theta;
            y_hat_te = trig_basis(x_te, m, omega) * theta;

            rmse_train(i) = calc_rmse(y_tr, y_hat_tr);
            rmse_test(i)  = calc_rmse(y_te, y_hat_te);
            r2_train(i)   = calc_r2(y_tr, y_hat_tr);
            r2_test(i)    = calc_r2(y_te, y_hat_te);

            fprintf('  %2d  |    %.5f    |    %.5f   |    %.5f   |    %.5f  |  %.3e\n', ...
                m, rmse_train(i), rmse_test(i), r2_train(i), r2_test(i), cond_P(i));
        end

        % Коэффициенты: theta = [a0, a1, b1, ..., am, bm]
        disp(' ');
        disp('Коэффициенты тригонометрического базиса:');
        for i = 1:numel(ms)
            m = ms(i);
            theta = thetas{i};
            fprintf('m = %d\n', m);
            fprintf('   a0 = %.5f\n', theta(1));
            for j = 1:m
                fprintf('   a%d = %.5f,  b%d = %.5f\n', j, theta(2*j), j, theta(2*j + 1));
            end
        end
        disp(' ');

        % Эталон: истинная f на тестовой выборке
        rmse_true = calc_rmse(y_te, f(x_te));
        r2_true   = calc_r2(y_te, f(x_te));
        fprintf('Истинная f на тестовой выборке: RMSE = %.5f, R^2 = %.5f\n', rmse_true, r2_true);

        figure;

        subplot(1, 2, 1);
        hold on;
        grid on;
        h_tr = plot(ms, rmse_train, 'o-', 'LineWidth', 1.3);
        h_te = plot(ms, rmse_test,  's-', 'LineWidth', 1.3);
        h_nz = yline(rmse_true, '--k', 'LineWidth', 1.3);
        xlabel('Число гармоник m');
        ylabel('RMSE');
        title('RMSE');
        legend([h_tr, h_te, h_nz], {'Обучающая', 'Тестовая', 'Истинная f'}, 'Location', 'best');
        hold off;

        subplot(1, 2, 2);
        hold on;
        grid on;
        h_tr = plot(ms, r2_train, 'o-', 'LineWidth', 1.3);
        h_te = plot(ms, r2_test,  's-', 'LineWidth', 1.3);
        h_nz = yline(r2_true, '--k', 'LineWidth', 1.3);
        xlabel('Число гармоник m');
        ylabel('R^2');
        title('Коэффициент детерминации R^2');
        legend([h_tr, h_te, h_nz], {'Обучающая', 'Тестовая', 'Истинная f'}, 'Location', 'best');
        hold off;

        sgtitle(sprintf('Тригонометрический базис: %s', samples.name));
        disp(' ');
    end

    function [theta, cond_P] = fit_trigonometric(x, y, m, omega)
        n = numel(x);
        theta = zeros(2*m + 1, 1);
        P = 1e-4 * eye(2*m + 1);
        Phi = trig_basis(x, m, omega);

        % Рекуррентный МНК, как в первой задаче
        for k = 1:n
            F = Phi(k, :);
            P = P + F' * F;
            residual = y(k) - F * theta;
            theta = theta + P \ (F' * residual);
        end

        cond_P = cond(P);
    end

    function Phi = trig_basis(x, m, omega)
        x = x(:);
        Phi = zeros(numel(x), 2*m + 1);

        % Столбцы: 1/2 (при a0), затем cos(j*omega*x) и sin(j*omega*x) для j = 1..m
        Phi(:, 1) = 0.5;
        for j = 1:m
            Phi(:, 2*j)     = cos(j * omega * x);
            Phi(:, 2*j + 1) = sin(j * omega * x);
        end
    end

    function r = calc_rmse(y, y_hat)
        r = sqrt(mean((y(:) - y_hat(:)).^2));
    end

    function r2 = calc_r2(y, y_hat)
        y = y(:);
        y_hat = y_hat(:);
        r2 = 1 - sum((y - y_hat).^2) / sum((y - mean(y)).^2);
    end

    function [theta, cond_P] = fit_polynomial(x, y, m)
        n = numel(x);
        theta = zeros(m + 1, 1);
        P = 1e-4 * eye(m + 1);

        for k = 1:n
            F = x(k) .^ (0:m);
            P = P + F' * F;
            residual = y(k) - F * theta;
            theta = theta + P \ (F' * residual);
        end

        cond_P = cond(P);
    end

    function [x_train, x_test] = generate_samples_with_noise(f, N, sigma, x0_train, x0_test, seed)
        rng(seed);
        x_train = generate_series_noisy(f, x0_train, sigma, N);

        rng(seed + 100);
        x_test = generate_series_noisy(f, x0_test, sigma, N);
    end

    function [x_train, x_test] = generate_samples_without_noise(f, N, x0_train, x0_test)
        x_train = generate_series_clean(f, x0_train, N);
        x_test  = generate_series_clean(f, x0_test, N);
    end

    function x = generate_series_noisy(f, x0, sigma, N)
            x = zeros(N + 1, 1);
            x(1) = x0;
            w = sigma * randn(N, 1);

            for k = 1:N
                x(k + 1) = f(x(k)) + w(k);
                check_divergence(x(k + 1), k);
            end
    end

    function x = generate_series_clean(f, x0, N)
            x = zeros(N + 1, 1);
            x(1) = x0;
                        
            for k = 1:N
                x(k + 1) = f(x(k));
                check_divergence(x(k + 1), k);
            end
    end

    function check_divergence(value, k)
            if ~isfinite(value) || abs(value) > 1e3
                error('Траектория разошлась на шаге k = %d. Измените x0 или seed.', k);
            end
    end
end
