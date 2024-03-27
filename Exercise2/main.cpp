#include "Eigen/Eigen"
#include <math.h>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

VectorXd fattorPALU(const MatrixXd& A,const VectorXd& b)
{
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}
VectorXd fattorQR(const MatrixXd& A,const VectorXd& b)
{
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}
double relErr (const VectorXd& sol,const VectorXd& x_approx)
{
    return (sol-x_approx).norm()/ sol.norm();
}
int main()
{
    //Inizializzo x come soluzione dei sistemi lineari.
    Vector2d sol;
    sol << -1.0e+0, -1.0e+00;

    // Inizializzo le coppie di matrici e termini noti, poi chiamo le funzioni che risolvono il sistema lineare con le diverse fattorizzazioni.
    // Infine chiamo la funzione che mi calcola gli errori relativi per i tre sistemi lineari.
    Matrix2d A;
    A << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d a;
    a << -5.169911863249772e-01, 1.672384680188350e-01;

    cout << "Relative error of linear system Ax=a in PALU fattorization is: " << relErr(sol,fattorPALU(A,a)) << endl;

    cout << "Relative error of linear system Ax=a in QR fattorization is: " << relErr(sol,fattorQR(A,a)) << endl;

    Matrix2d B;
    B << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b;
    b << -6.394645785530173e-04, 4.259549612877223e-04;


    cout << "Relative error of linear system Bx=b in PALU fattorization is: " << relErr(sol,fattorPALU(B,b)) << endl;


    cout << "Relative error of linear system Bx=b in QR fattorization is: " << relErr(sol,fattorQR(B,b)) << endl;

    Matrix2d C;
    C << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d c;
    c << -6.400391328043042e-10, 4.266924591433963e-10;


    cout << "Relative error of linear system Cx=c in PALU fattorization is: " << relErr(sol,fattorPALU(C,c)) << endl;


    cout << "Relative error of linear system Cx=c in QR fattorization is: " << relErr(sol,fattorQR(C,c)) << endl;


    return 0;
}
