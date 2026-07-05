import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import time
import torch
import torch.nn as nn



class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),

            nn.Linear(32,32),
            nn.Tanh(),

            nn.Linear(32,32),
            nn.Tanh(),

            nn.Linear(32,1)
        )

    def forward(self,t):
        return self.net(t)
#Configurar el modo de redondeo en las funciones
cantidad_decimales = 0.01
decimales = Decimal(str(cantidad_decimales))

#Semilla para el Random
semilla = 42
np.random.seed(semilla)


def metodo_e(x_0, v_0, h, w, N, printear = False):
    x_n = x_0
    v_n = v_0
    x=[]
    x.append(x_n)
    for i in range(N-1):
        if printear:
            print(f"x_{i+1+1} = {x_n}")
        v_n = Decimal(str(v_n)) - Decimal(str(x_n))*Decimal(str(h))*Decimal(str(w))**2
        x_n = Decimal(str(x_n)) + Decimal(str(h))*Decimal(str(v_n))
        x.append(x_n)
    return x

def finite_diffe(x_0, v_0, h, w, N):
    x_n0=Decimal(str(x_0))
    x_n=x_n0-Decimal(str(h))*Decimal(str(v_0))-((Decimal(str(h))*Decimal(str(w)))**2)/Decimal(str(2))
    x=[]
    x.append(x_n0)
    x.append(x_n)
    for i in range(N-2):
        x_n=(Decimal(str(2))-(Decimal(str(w))*Decimal(str(h)))**2)*Decimal(str(x_n))-x[i]
        x.append(x_n)
    return x

def error_e(x_lista, y_lista, N):
    err=[]
    l2error = 0
    error_max = 0
    for i in range(N):
        er=np.abs(x_lista[i]-y_lista[i])
        er2 = er**2
        l2error += er2
        if er > error_max:
            error_max = er
        err.append(er)
    l2error = np.sqrt(Decimal(str(1/N))* l2error)
    return err, error_max, l2error

N = 500
h = 0.1
w = np.pi/2
t_fin = h * N

inicio_contador = time.time()
x_list=metodo_e(1, 0, h, w, N)
fin_contador = time.time()
print(f"Tiempo: {fin_contador - inicio_contador}")

t=np.linspace(0,t_fin,N)
y_list=[]

for i in range(N):
    y_list.append(Decimal(str(np.cos(w*t[i]))))

lista_err_e, error_max_e, l2error_e = error_e(x_list, y_list, N)
plt.plot(t,x_list,'-g')
plt.plot(t,y_list,'-r')
plt.title("Comparación Metodo euler Valor real")
plt.show()

inicio_contador2 = time.time()
x_dif=finite_diffe(1,0,h,w,N)
fin_contador2 = time.time()
print(f"Tiempo: {fin_contador2 - inicio_contador2}")

lista_err_d, error_max_d, l2error_d=error_e(x_dif,y_list,N)
plt.plot(t,x_dif,'-b')
plt.plot(t,y_list,'-r')
plt.title("Comparacion Dif finitas vs Valor real")
plt.show()

print("Error maximo: " + str(error_max_e))
print("Error l2 :"+ str(l2error_e))
plt.plot(t,lista_err_d,'-r')
plt.plot(t,lista_err_e,'-c')
plt.show()

model=PINN()
T=h*N
lr=1e-3
trains=5000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#Puntos a entrenar
t_train=torch.linspace(0, T, N).reshape(-1, 1)
#Forzar derivabilidad
t_train.requires_grad_(True)

loss_history=[]
iniciopinn=time.time()
for _ in range(trains):

    optimizer.zero_grad()
    #Entrena fuafuafua
    x=model(t_train)

    x_t=torch.autograd.grad(
        x, t_train,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    x_tt = torch.autograd.grad(
        x_t, t_train,
        grad_outputs=torch.ones_like(x_t),
        create_graph=True
    )[0]
    #residuo fisico
    res=x_tt+(w**2)*x
    loss_fis=torch.mean(res**2)

    t0=torch.tensor([[0.0]], requires_grad=True)
    x0=model(t0)
    x0_t= torch.autograd.grad(
        x0,t0,
        grad_outputs=torch.ones_like(x0),
        create_graph=True
    )[0]

    loss_ic=(x0-1.0)**2+(x0_t-0.0)**2

    loss=loss_fis+loss_ic
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if _ % 500 == 0:
        print(f"Epoch {_:5d} | Loss = {loss.item():.8f}")

t_test= torch.linspace(0,T,N).reshape(-1,1)

with torch.no_grad():
    x_pred=model(t_test).numpy()
fincontpinn=time.time()
t_np=t_test.numpy()
x_real=np.cos(t_np)
print(f"Tiempo PINN: {fincontpinn-iniciopinn}")
error_max=np.max(np.abs(x_pred-x_real))
print("Error maximo:", error_max)

error_l2=np.sqrt(np.mean((x_pred-x_real)**2))
print("Error L2:", error_l2)

# Gráfico solución
plt.figure(figsize=(10,5))
plt.plot(t_np, x_real, label="Exacta cos(t)", linewidth=2)
plt.plot(t_np, x_pred, "--", label="PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("PINN vs Solución Exacta")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico error
plt.figure(figsize=(10,5))
plt.plot(t_np, np.abs(x_pred - x_real))
plt.xlabel("t")
plt.ylabel("Error absoluto")
plt.title("Error PINN")
plt.grid(True)
plt.show()

# Gráfico loss
plt.figure(figsize=(10,5))
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergencia entrenamiento")
plt.grid(True)
plt.show()

#Comparacion de costo computacional
tiempos = {
    'Euler': fin_contador - inicio_contador,
    'Dif. Finitas': fin_contador2 - inicio_contador2,
    'PINN': fincontpinn - iniciopinn
}
plt.figure(figsize=(8, 5))
bars = plt.bar(tiempos.keys(), tiempos.values(), color=['green', 'blue', 'orange'])
plt.yscale('log') # Escala logarítmica porque la PINN suele tardar órdenes de magnitud más
plt.ylabel('Tiempo de ejecución (segundos) - Escala Log')
plt.title('Comparación de Costo Computacional')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.6f}s', ha='center', va='bottom', fontsize=10)
plt.show()

# 2. Sensibilidad a condiciones iniciales (x0 = 1.0)
variaciones_x0 = [0.8, 1.0, 1.2] 
v_0 = 0.0

plt.figure(figsize=(12, 5))

for idx, x0_var in enumerate(variaciones_x0): #Solución para cada x0 
    x_real_var = x0_var * np.cos(w * t_np)
    #Euler
    x_euler_var = metodo_e(x0_var, v_0, h, w, N)
    x_euler_np = np.array([float(val) for val in x_euler_var]) # Convertir Decimal a float para restar
    err_euler = np.abs(x_euler_np - x_real_var.flatten())
   
    #Diferencias Finitas
    x_dif_var = finite_diffe(x0_var, v_0, h, w, N)
    x_dif_np = np.array([float(val) for val in x_dif_var])
    err_dif = np.abs(x_dif_np - x_real_var.flatten())
    
    #PINN
    model_var = PINN()
    optimizer_var = torch.optim.Adam(model_var.parameters(), lr=1e-3)
    for epoch in range(2000):
        optimizer_var.zero_grad()
        x_p = model_var(t_train)
        x_t = torch.autograd.grad(x_p, t_train, grad_outputs=torch.ones_like(x_p), create_graph=True)[0]
        x_tt = torch.autograd.grad(x_t, t_train, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]
        loss_fis = torch.mean((x_tt + (w**2) * x_p)**2)       
        x0_p = model_var(t0)
        x0_t = torch.autograd.grad(x0_p, t0, grad_outputs=torch.ones_like(x0_p), create_graph=True)[0]
        loss_ic = (x0_p - x0_var)**2 + (x0_t - v_0)**2
        loss = loss_fis + loss_ic
        loss.backward()
        optimizer_var.step()
        
    with torch.no_grad():
        x_pinn_pred = model_var(t_test).numpy()
    err_pinn = np.abs(x_pinn_pred.flatten() - x_real_var.flatten())
    
    #Gráfico del Error Absoluto Máximo o Medio
    plt.subplot(1, 3, idx+1)
    plt.plot(t_np, err_euler, label='Euler', color='green', alpha=0.7)
    plt.plot(t_np, err_dif, label='Dif. Finitas', color='blue', alpha=0.7)
    plt.plot(t_np, err_pinn, label='PINN', color='orange', linestyle='--')
    plt.title(f'Errores para $x_0 = {x0_var}$')
    plt.xlabel('t')
    if idx == 0:
        plt.ylabel('Error Absoluto')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()