//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
#include <chrono>

using namespace sycl;

struct meta_Data {
    int num_element;
    int array[100]; // Assuming a fixed-size array for simplicity
};


// Optimized implementation of Bubble sort

using namespace std;

// An optimized version of Bubble Sort
void bubbleSort(int* arr, int n)
{
    int i, j;
    bool swapped;
    for (i = 0; i < n - 1; i++) {
        swapped = false;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        // If no two elements were swapped
        // by inner loop, then break
        if (swapped == false)
            break;
    }
}




// A utility function to swap two elements 
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function is same in both iterative and recursive*/
int partition(int arr[], int l, int h)
{
    int x = arr[h];
    int i = (l - 1);

    for (int j = l; j <= h - 1; j++) {
        if (arr[j] <= x) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[h]);
    return (i + 1);
}

/* A[] --> Array to be sorted,
l --> Starting index,
h --> Ending index */
void quickSortIterative(int arr[], int l, int h)
{
    // Create an auxiliary stack 
    int stack[100];
    //int* stack = malloc_shared<int>(h - l + 1, q);

    // initialize top of stack 
    int top = -1;

    // push initial values of l and h to stack 
    stack[++top] = l;
    stack[++top] = h;

    // Keep popping from stack while is not empty 
    while (top >= 0) {
        // Pop h and l 
        h = stack[top--];
        l = stack[top--];

        // Set pivot element at its correct position 
        // in sorted array 
        int p = partition(arr, l, h);

        // If there are elements on left side of pivot, 
        // then push left side to stack 
        if (p - 1 > l) {
            stack[++top] = l;
            stack[++top] = p - 1;
        }

        // If there are elements on right side of pivot, 
        // then push right side to stack 
        if (p + 1 < h) {
            stack[++top] = p + 1;
            stack[++top] = h;
        }
    }
}

#include <CL/sycl.hpp>
#include <algorithm> // for std::min
#include <iostream>

using namespace sycl;

/* Function to merge the two halves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(sycl::buffer<int, 1>& arr, int l, int m, int r) {
    auto accessor = arr.get_access<sycl::access::mode::read_write>();

    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary buffers for the left and right subarrays
    int* L = new int[n1];
    int* R = new int[n2];

    // Copy data to temp arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = accessor[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = accessor[m + 1 + j];

    // Merge the temp arrays back into arr[l..r]
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            accessor[k] = L[i];
            i++;
        }
        else {
            accessor[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        accessor[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        accessor[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

void mergeSort(sycl::buffer<int, 1>& arr, int n) {
    for (int curr_size = 1; curr_size <= n - 1; curr_size *= 2) {
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = std::min(left_start + curr_size - 1, n - 1);
            int right_end = std::min(left_start + 2 * curr_size - 1, n - 1);

            merge(arr, left_start, mid, right_end);
        }
    }
}



int main() {
    int N = 7; // Adjust the size of the array as needed

    queue q;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    int* arr = malloc_shared<int>(N, q);
    int* arr2 = malloc_shared<int>(N, q);
    for (int i = 0; i < N; ++i) arr[i] = rand() % 100; // Initialize the array with random values
    //for (int i = 0; i < N; ++i) arr2[i] = rand() % 100;
    for (int i = 0; i < N; ++i) arr2[i] = arr[i];
    q.parallel_for(range<1>(N), [=](id<1> i) {

        bubbleSort(arr, N);
        }).wait();

        for (int i = 0; i < N; ++i) std::cout << arr[i] << " ";
        std::cout << std::endl;

        printf("Hello>\n");
    int temp[] = { 4, 3, 5, 11, 3, 2, 3 };
    int n = 7;
    int* arr3 = malloc_shared<int>(n, q);
    //for (int i = 0; i < n; ++i) arr3[i] = temp[i];
    for (int i = 0; i < n; ++i) arr3[i] = arr2[i];

    q.parallel_for(range<1>(n), [=](id<1> i) {

        quickSortIterative(arr3, 0, n - 1);
        }).wait();
        for (int i = 0; i < n; ++i) std::cout << arr3[i] << " ";
    printf("Hello>\n");
    
    //Merge sort

    constexpr int N_merge = 10;
    int arrMerge[N_merge] = { 12, 11, 13, 5, 6, 7, 9, 8, 1, 2 };
    sycl::queue queue(sycl::gpu_selector{});

    // Create buffer for the array
    sycl::buffer<int, 1> arrBuffer(arrMerge, sycl::range<1>(N_merge));
    mergeSort(arrBuffer, N_merge);
    
    auto result = arrBuffer.get_access<sycl::access::mode::read>();

    // Print sorted data
    for (int i = 0; i < N_merge; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;



    sycl::free(arr, q);
    sycl::free(arr2, q);
    return 0;
}