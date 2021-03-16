import math

import matplotlib.pyplot as plt
import numpy as np

# Zu lösendes LGS aus Aufgabe
A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]])

# Hilfsmatritzen zur Berechnung der Dreicksmatritzen
L = np.array([[0, 0, 0, 0],
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0]])

R = np.array([[0, 1, 1, 1],
              [0, 0, 1, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])

# Berechnung Diagonalmatrix
Dm = np.diag(np.diag(A, 0))

# Dreiecksmatrix Links und Rechts
Lm = A * L
Rm = A * R

# Lösung aus Aufgabe
b = np.array([6, 25, -11, 15])

# Anfangspunkt für Iteration
x = np.array([0, 0, 0, 0])

# Toleranz aus Aufgabenstellung
tol = 0.001


def jacobi(A, b, x, Dm, Lm, Rm, tol, maxiter=200):
    iteration = 1
    x0 = x.copy()
    # Berechnung Jacobimatrix
    Jm = np.dot(-np.linalg.inv(Dm), (Lm + Rm))

    yIterat = []
    x1lJ = []
    x2lJ = []
    x3lJ = []
    x4lJ = []

    while iteration <= maxiter:
        # Berechnung Fixpunkt
        c = np.dot(np.linalg.inv(Dm), b)
        x = np.dot(Jm, x0) + c
        # Vorbereitung zur Untersuchung ob Abbruchkriterium erreicht
        var = abs(x0 - x)
        # Prüfung ob Toleranzgrenze erreicht ist
        if var[0] < tol and var[1] < tol and var[2] < tol and var[3] < tol:
            break;
        # Wert vorgehenden Iteration sichern
        x0 = x.copy()
        # Erzeugen einer Liste zum Plotten des Konvergenzverhaltens
        x1lJ.append(x0[0])
        x2lJ.append(x0[1])
        x3lJ.append(x0[2])
        x4lJ.append(x0[3])
        yIterat.append(iteration)
        iteration = iteration + 1
    iteration = iteration -1
    print("Jacobi Verfahren")
    print("Lösung gefunden nach: " + str(iteration) + " Iterationen")

    print("x1 = " + str(x0[0]))
    print("x2 = " + str(x0[1]))
    print("x3 = " + str(x0[2]))
    print("x4 = " + str(x0[3]))

    # Plotten der Konvergenz
    p1 = plt.plot([yIterat], [x1lJ], "ro")
    p2 = plt.plot([yIterat], [x2lJ], "go")
    p3 = plt.plot([yIterat], [x3lJ], "bo")
    p4 = plt.plot([yIterat], [x4lJ], "yo")

    plt.axis([0, 10, -2, 3])
    plt.title('Jacobi Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Wert')
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('x1', 'x2', 'x3', 'x4'))
    plt.show()


def gauss_seidel(A, b, x, Dm, Lm, Rm, tol, maxiter=200):
    iteration = 1
    yIterat = []
    x1lgs = []
    x2lgs = []
    x3lgs = []
    x4lgs = []
    x0 = x.copy()
    # Berechnung Gauß-Seidel-Matrix h1
    gsM = np.dot(-np.linalg.inv((Dm + Lm)), Rm)
    while iteration <= maxiter:

        # Berechnung Iterationsmatrix
        x = np.dot(gsM, x0) + np.dot(np.linalg.inv(Dm + Lm), b)
        # Vorbereitung zur Untersuchung ob Abbruchkriterium erreicht
        var = abs(x0 - x)
        # Prüfung ob Toleranzgrenze erreicht ist
        if var[0] < tol and var[1] < tol and var[2] < tol and var[3] < tol:
            break;
        # Sichern vorheriger Iterationsschritt
        x0 = x.copy()
        # Erzeugen einer Liste zum Plotten des Konvergenzverhaltens
        x1lgs.append(x0[0])
        x2lgs.append(x0[1])
        x3lgs.append(x0[2])
        x4lgs.append(x0[3])
        yIterat.append(iteration)
        iteration = iteration + 1
    iteration = iteration - 1
    print("Gauß-Seidel Verfahren")
    print("Lösung gefunden nach: " + str(iteration) + " Iterationen")

    print("x1 = " + str(x0[0]))
    print("x2 = " + str(x0[1]))
    print("x3 = " + str(x0[2]))
    print("x4 = " + str(x0[3]))

    # Plotten der Konvergenz
    p1 = plt.plot([yIterat], [x1lgs], "ro")
    p2 = plt.plot([yIterat], [x2lgs], "go")
    p3 = plt.plot([yIterat], [x3lgs], "bo")
    p4 = plt.plot([yIterat], [x4lgs], "yo")

    plt.axis([0, 5, -2, 3])
    plt.title('Gauß-Seidel')
    plt.xlabel('Iteration')
    plt.ylabel('Wert')
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('x1', 'x2', 'x3', 'x4'))
    plt.show()


def SOR(A, b, x, Dm, Lm, Rm, tol, maxiter=100):
    iteration = 1
    yIterat = []
    x1lgs = []
    x2lgs = []
    x3lgs = []
    x4lgs = []
    x0 = x.copy()
    # Berechnung p aus Jacobi-Matrix
    Jm = np.dot(-np.linalg.inv(Dm), (Lm + Rm))
    p = np.linalg.norm(Jm, 2)
    print("Spektralradius der Jacobi-Matrix: " +str(p))
    omega = omega_opt(p)
    print("Ideales Omega berechnet: " + str(omega))
    #Berechnung SOR-Matrix hw
    Hw = np.dot(np.linalg.inv((Dm + (omega * Lm))), ((1 - omega) * Dm - omega * Rm))
    while iteration <= maxiter:
        # Berechnung Iterationsmatrix
        x = np.dot(Hw, x0) + np.dot(omega * (np.linalg.inv(Dm + (omega * Lm))), b)
        # Vorbereitung zur Untersuchung ob Abbruchkriterium erreicht
        var = abs(x0 - x)
        # Prüfung ob Toleranzgrenze erreicht ist
        if var[0] < tol and var[1] < tol and var[2] < tol and var[3] < tol:
            break;

        x0 = x.copy()
        # Erzeugen einer Liste zum Plotten des Konvergenzverhaltens
        x1lgs.append(x0[0])
        x2lgs.append(x0[1])
        x3lgs.append(x0[2])
        x4lgs.append(x0[3])
        yIterat.append(iteration)
        iteration = iteration + 1
    iteration = iteration - 1
    print("Lösungswerte")
    print("x1 = " + str(x0[0]))
    print("x2 = " + str(x0[1]))
    print("x3 = " + str(x0[2]))
    print("x4 = " + str(x0[3]))
    print("Lösung gefunden nach Iterationen: " + str(iteration))
    # Plotten der Konvergenz
    p1 = plt.plot([yIterat], [x1lgs], "ro")
    p2 = plt.plot([yIterat], [x2lgs], "go")
    p3 = plt.plot([yIterat], [x3lgs], "bo")
    p4 = plt.plot([yIterat], [x4lgs], "yo")

    plt.axis([0, 5, -2, 3])
    plt.title('SOR')
    plt.xlabel('Iteration')
    plt.ylabel('Wert')
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('x1', 'x2', 'x3', 'x4'))
    plt.show()


def omega_opt(p):
    # result = 2 / (1 + math.sqrt(1 - p**2))
    #result = 1 + (p / (1 + math.sqrt(1 - p ** 2))) ** 2
    result = (2*(1 - math.sqrt(1-p**2))) / p**2
    return result

print("Willkommen zum Numerik Script zum interativen Lösen von LGS")
print("Folgende Eingaben sind möglich:")
print("1 für Jacobi")
print("2 für Gauß-Seidel")
print("3 für SOR")

userInput = input("Eingabe int 1-3: ")

if userInput == '1':
    jacobi(A, b, x, Dm, Lm, Rm, tol)

if userInput == '2':
    gauss_seidel(A, b, x, Dm, Lm, Rm, tol)

if userInput == '3':
    SOR(A, b, x, Dm, Lm, Rm, tol)


