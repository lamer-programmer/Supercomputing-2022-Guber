#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>


double GetScalarProduct(const std::vector<double> & v1, const std::vector<double> & v2)
{
    double product = 0.0;

    for (int i = 0; i < v1.size(); i++)
    {
        product += v1[i] * v2[i];
    }

    return product;
}

double GetSquaredNorm(const std::vector<double> & vec)
{
    double norm = 0.0;

    for (int i = 0; i < vec.size(); i++)
    {
        norm += vec[i] * vec[i];
    }

    return norm;
}

void SetTest(
    std::vector<double> & a, 
    std::vector<double> & b, 
    std::vector<double> & x, 
    int displs, int localSize, int taskSize
)
{
    for (int i = 0; i < localSize; i++)
    {
        for (int j = 0; j < taskSize; j++)
        {
            if (displs + i == j)
            {
                a[i * taskSize + j] = 2.0;
            } else
            {
                a[i * taskSize + j] = 1.0;
            }
        }

        b[i] = taskSize + 1;
        x[i] = 0.0;
    }
}

void dgemv(
    const std::vector<double> & mat,
    const std::vector<double> & vec,
    std::vector<double> & result
)
{
    int m = vec.size();
    int n = mat.size() / m;

    for (int i = 0; i < n; i++)
    {
        result[i] = 0;

        for (int j = 0; j < m; j++)
        {
            result[i] += mat[i * m + j] * vec[j];
        }
    }
}

void PrintVector(const std::vector<double> & vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << " ";
    }

    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    auto startTime = MPI_Wtime();

    const int taskSize = 1000;
    const double epsilon = 1e-10;
    const int maxIterations = 10000;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // вектор с размерами кусков векторов и матриц для каждого процесса
    std::vector<int> localSizes(size, 0.0);
    for (int i = 0; i < size; i++)
    {
        localSizes[i] = taskSize / size + ((i < taskSize % size) ? 1 : 0);
    }

    // вектор со смещениями для каждого процесса
    std::vector<int> displs(size, 0.0);
    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + localSizes[i - 1];
    }

    const int localSize = localSizes[rank];

    std::vector<double> b(taskSize, 0.0);
    std::vector<double> x(taskSize, 0.0);

    std::vector<double> bLocal(localSize, 0.0);
    std::vector<double> xLocal(localSize, 0.0);
    std::vector<double> aLocal(taskSize * localSize, 0.0);

    // ставим тест и собираем вектора по кусочкам из каждого процесса
    SetTest(aLocal, bLocal, xLocal, displs[rank], localSize, taskSize);
    MPI_Allgatherv(
        xLocal.data(), localSize, MPI_DOUBLE, x.data(),
        localSizes.data(), displs.data(),
        MPI_DOUBLE, MPI_COMM_WORLD
    );

    MPI_Allgatherv(
        bLocal.data(), localSize, MPI_DOUBLE, b.data(),
        localSizes.data(), displs.data(),
        MPI_DOUBLE, MPI_COMM_WORLD
    );

    std::vector<double> yLocal(localSize, 0.0);
    std::vector<double> y(taskSize, 0.0);
    // вектор для промежуточных вычислений - перемножений матриц
    std::vector<double> tempVector(localSize, 0.0);

    auto normB = GetSquaredNorm(b);
    double cond = 0.0;
    int iterations = 0;

    // умножаем кусок матрицы А на вектор х
    dgemv(aLocal, x, tempVector);

    for (int i = 0; i < localSize; i++)
    {
        yLocal[i] = tempVector[i] - bLocal[i];
    }

    MPI_Allgatherv(
        yLocal.data(), localSize, MPI_DOUBLE, y.data(),
        localSizes.data(), displs.data(),
        MPI_DOUBLE, MPI_COMM_WORLD
    );

    cond = GetSquaredNorm(y) / normB;

    while (cond > epsilon && iterations++ < maxIterations)
    {
        dgemv(aLocal, y, tempVector);

        auto localNumerator = GetScalarProduct(yLocal, tempVector);
        auto localDenominator = GetScalarProduct(tempVector, tempVector);

        double numerator = 0.0;
        double denominator = 0.0;

        MPI_Allreduce(&localNumerator, &numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localDenominator, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double tau = numerator / denominator;

        for (int i = 0; i < localSize; i++) 
        {
            xLocal[i] -= tau * yLocal[i];
        }

        if (rank == 0 && iterations == 2)
        {
            std::cout << "tau = " << tau << std::endl;
        }

        MPI_Allgatherv(
            xLocal.data(), localSize, MPI_DOUBLE, x.data(),
            localSizes.data(), displs.data(),
            MPI_DOUBLE, MPI_COMM_WORLD
        );
        
        // умножаем кусок матрицы А на вектор х
        dgemv(aLocal, x, tempVector);

        for (int i = 0; i < localSize; i++)
        {
            yLocal[i] = tempVector[i] - bLocal[i];
        }

        MPI_Allgatherv(
            yLocal.data(), localSize, MPI_DOUBLE, y.data(),
            localSizes.data(), displs.data(),
            MPI_DOUBLE, MPI_COMM_WORLD
        );

        // обновляем условие остановки
        cond = GetSquaredNorm(y) / normB;
    }

    if (rank == 0)
    {
        std::cout << std::endl << "Elapsed time " << MPI_Wtime() - startTime << std::endl;
        PrintVector(x);
    }

    MPI_Finalize();
}
