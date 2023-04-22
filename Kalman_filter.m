

A = [2 1 0; 0 2 1; 0 0 2]

C = [4,0,0]

V = 1

W = [1,0,0;0,1,0;0,0,1]

x = transpose(mvnrnd([0,0,0],[1,0,0;0,1,0;0,0,1]))

x_0 = x

x = [0;0;0]

sigma_t = [1,0,0;0,1,0;0,0,1]

m_tilde = [0;0;0]

x_vals0 = [];
x_vals1 = [];
x_vals2 = [];
m_vals1 = [];
m_vals2 = [];
m_vals3 = [];
x_vals0(1) = x(1);
x_vals1(1) = x(2);
x_vals2(1) = x(3);
m_vals1(1) = m_tilde(1);
m_vals2(1) = m_tilde(2);
m_vals3(1) = m_tilde(3);

%sigma_t = calc_sigma(x,sigma_t,A,C,W,V)

test = [5,2,0;2,5,2;0,2,4] + [1 0 0;0 1 0; 0 0 1] - [8;0;0]*(1/17)*[8,0,0]



for i = 1:1000

    x = calculate_state(x)
    sigma_t = calc_sigma(x,sigma_t,A,C,W,V)
    m_tilde = calc_m_tilde(m_tilde,sigma_t,x,A,C,V)

    x_vals0(i) = x(1);
    x_vals1(i) = x(2);
    x_vals2(i) = x(3);

    m_vals1(i) = m_tilde(1);
    m_vals2(i) = m_tilde(2);
    m_vals3(i) = m_tilde(3);


end


diff1 = x_vals0 - m_vals1
diff2 = x_vals1 - m_vals2
diff3 = x_vals2 - m_vals3

error1 = abs((m_vals1 - x_vals0)/x_vals0) * 100
error2 = abs((m_vals2 - x_vals1)/x_vals1) * 100
error3 = abs((m_vals3 - x_vals2)/x_vals2) * 100

p_error_1_10 = abs((m_vals2(10)-x_vals1(10))/x_vals1(10)) * 100

p_error_2 = abs((m_vals3(1000)-x_vals2(1000))/x_vals2(1000)) * 100

%observ = obsv(A,C)

x_axis = 1:1000;

figure()
subplot(3,1,1)
plot(x_axis,x_vals0), legend('x1')

subplot(3,1,2)
plot(x_axis,m_vals1), legend('m1')

subplot(3,1,3)
plot(x_axis,x_vals0-m_vals1), legend('x1-m1')

figure()
subplot(3,1,1)
plot(x_axis,x_vals1), legend('x2')

subplot(3,1,2)
plot(x_axis,m_vals2), legend('m2')

subplot(3,1,3)
plot(x_axis,x_vals1-m_vals2), legend('x2-m2')


figure()
subplot(3,1,1)
plot(x_axis,x_vals2), legend('x3')

subplot(3,1,2)
plot(x_axis,m_vals3), legend('m3')

subplot(3,1,3)
plot(x_axis,x_vals2-m_vals3), legend('x3-m3')

%{
figure()
plot(x_axis,x_vals1,x_axis,m_vals2)

figure()
plot(x_axis,x_vals2,x_axis,m_vals3)


figure()
plot(x_axis,x_vals0 - m_vals1), legend('difference for state 1')

figure()
plot(x_axis,(x_vals1 -m_vals2)), legend('difference for state 2')

figure()
plot(x_axis,x_vals2-m_vals3), legend('difference for state 3')
%}


function m_tilde = calc_m_tilde(m_tilde,sigma_t,x,A,C,V)

    v_t = calculate_v_t()
    y = C*x + v_t
    m_tilde = A*m_tilde + sigma_t*transpose(C)*((C*sigma_t*transpose(C)+V)^-1)*(y-C*A*m_tilde)

end

function sigma_t =calc_sigma(x,sigma_t,A,C,W,V)
    
    sigma_t = A * sigma_t * transpose(A) + W - (A*sigma_t*transpose(C))*((C*sigma_t*transpose(C)+V)^-1)*(C*sigma_t*transpose(A))

end

function x = calculate_state(x)

    A = [2 1 0; 0 2 1; 0 0 2];
    w_t = calculate_w_t();
    x = A * x + w_t;
end

function w_t = calculate_w_t()
 w_t = transpose(mvnrnd([0,0,0],[1,0,0 ; 0,1,0 ; 0,0,1]));
end

function v_t = calculate_v_t()
    v_t = normrnd(0,1)
    
end

