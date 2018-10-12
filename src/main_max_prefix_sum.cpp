#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"



template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {

            auto max_n_tmp = static_cast<unsigned int>(n);
            std::vector<int> as_gpu(max_n_tmp*2);

            for(int i=0; i<max_n_tmp; i++){
                as_gpu[i+max_n_tmp] = as[i];
                as_gpu[i] = std::max(as[i], 0);
            }

            std::vector<int> res_gpu(max_n_tmp*2, 0);

            char *argvv[] = { "device", "0" };
            gpu::Device device = gpu::chooseGPUDevice(2, argvv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            const unsigned int workGroup = 128;
            unsigned count_group = (max_n_tmp + workGroup - 1) / workGroup;
            gpu::gpu_mem_32i buffer_tmp;

            buffer_tmp.resizeN(count_group * 2);

            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            kernel.compile();

            {
                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    max_n_tmp = static_cast<unsigned int>(n);

                    gpu::gpu_mem_32i buffer_old;
                    buffer_old.resizeN(max_n_tmp*2);

                    buffer_old.writeN(as_gpu.data(), max_n_tmp*2);

                    while (max_n_tmp > 1) {
                        kernel.exec(gpu::WorkSize(workGroup, max_n_tmp),
                                    buffer_old, max_n_tmp, buffer_tmp);

                        buffer_old.swap(buffer_tmp);
                        max_n_tmp = (max_n_tmp + workGroup - 1) / workGroup;

                    }
                    t.nextLap();

                    buffer_old.readN(res_gpu.data(), 2);
                    EXPECT_THE_SAME(reference_max_sum, std::max(res_gpu[0], 0), "GPU result should be consistent!");
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }

}
