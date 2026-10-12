omega = 1.32;
t_start = 0;
t_end = 20;
N = 10000;

t_eval = linspace(t_start, t_end, N);

data = step_a_generate_data(t_eval, omega);

derivatives = step_b_dx_calc(data, omega);

step_c_parameter_estimation(data, derivatives, omega);

step_d_noisy_estimation(data, derivatives, omega)

step_e_finite_difference_estimation(data, derivatives, omega)

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

function step_e_finite_difference_estimation(data, derivatives, omega)
    t = data(:, 1);
    X = data(:, 2:3);
    N_points = length(t);
    h = t(2) - t(1);

    dx_fd = zeros(N_points, 2);
    dx_fd(1:end-1, :) = (X(2:end, :) - X(1:end-1, :)) / h;
    dx_fd(end, :)     = (X(end, :) - X(end-1, :)) / h;

    disp('Step д):')
    disp('   t_k     |   dx1_fd(t_k)   |   dx2_fd(t_k)');
    disp([t(1:5), dx_fd(1:5, :)]);
    fprintf('h = %.6f\n', h);
    fprintf('Макс. отклонение dx2_fd от точной производной: %.4e\n', ...
        max(abs(dx_fd(:, 2) - derivatives(:, 2))));

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

