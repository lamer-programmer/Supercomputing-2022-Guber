#define _USE_MATH_DEFINES

#include <iostream>
#include <valarray>
#include <cmath>
#include <fstream>

double GetSquaredNorm(const std::valarray<double> & vec)
{
    double norm = 0.0;

    for (int i = 0; i < vec.size(); i++)
    {
        norm += vec[i] * vec[i];
    }

    return norm;
}

double GetScalarProduct(const std::valarray<double> & a, const std::valarray<double> & b)
{
    double product = 0.0;

    for (int i = 0; i < a.size(); i++)
    {
        product += a[i] * b[i];
    }

    return product;
}

// mat - n * m
std::valarray<double> dgemv(
    const std::valarray<double> & mat, 
    const std::valarray<double> & vec, 
    int n, int m
)
{
    std::valarray<double> result(0.0, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            result[i] += mat[i * m + j] * vec[j];
        }
    }

    return result;
}

void SaveVector(const std::valarray<double> & vec, const std::string & filename)
{
    std::ofstream out(filename);

    for (int i = 0; i < vec.size(); i++)
    {
        out << vec[i] << " ";
    }

    out << std::endl;
}

// fill matrix A and vector b
void SetTest1(std::valarray<double> & A, std::valarray<double> & b)
{
    int size = b.size();
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                A[i * size + j] = 2.0;
            } else
            {
                A[i * size + j] = 1.0;
            }
        }

        b[i] = size + 1;
    }
}

void SetTest2(std::valarray<double> & A, std::valarray<double> & b)
{
    int size = b.size();
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                A[i * size + j] = 2.0;
            } else
            {
                A[i * size + j] = 1.0;
            }
        }
    }

    std::valarray<double> u(size);
    for (int i = 0; i < size; i++)
    {
        u[i] = std::sin(2 * i * M_PI / size);
    }
    
    b = dgemv(A, u, size, size);
}

int main()
{
    const int size = 50;
    const int maxIter = 10000000;
    // epsilon возведён в квадрат, чтобы не считать лишний раз корень
    const double epsilon = 1e-10;

    std::valarray<double> A(0.0, size * size);
    std::valarray<double> b(0.0, size);
    std::valarray<double> x(0.0, size);
    std::valarray<double> y(size);
    std::valarray<double> tempVector(size);

    SetTest2(A, b);

    double tau = 0.0;
    double norm = GetSquaredNorm(b);

    int iter = 0;

    tempVector = dgemv(A, x, size, size);

    for (int i = 0; i < maxIter; i++)
    {
        y = tempVector - b;
        tempVector = dgemv(A, y, size, size);
        tau = GetScalarProduct(y, tempVector) / GetSquaredNorm(tempVector);
        x = x - tau * y;

        tempVector = dgemv(A, x, size, size);
        if (GetSquaredNorm(tempVector - b) / norm < epsilon)
        {
            break;
        }
    }

    SaveVector(x, "naive-test1.dat");
}
