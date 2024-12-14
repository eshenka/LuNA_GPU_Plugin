#include "gpu_thread_pool.h"

#include <cassert>
#include <sched.h>
#include <sstream>

#include "gpu_timer.h"
#include "common.h"
#include "ucenv.h"
#include <cuda_runtime.h>

GPUThreadPool::GPUThreadPool(const Config *config)
    : on_empty_handler_([](){}), on_submit_handler_([](){}), running_jobs_(0), stop_flag_(false), conf(config)
{
    stop();
    set_gpu_related_params(config);
}

GPUThreadPool::~GPUThreadPool()
{
    stop();
}

void GPUThreadPool::set_gpu_related_params(const Config *config)
{
    this->use_gpu_ = config->get_bool_config("--use-gpu");
    // this->gpu_id_list_ = config->get_string_config("--gpu-dev-ids=");
    this->gpu_build_dir_ = config->get_build_dir();
    this->use_gpu_timer_ = config->get_bool_config("--gpu-use-timer");

    // if (this->use_gpu_)
    // {
    //     const char *list = this->gpu_id_list_.c_str();
    //     printf("[GPU] Device list: %s\n", list);
    //     int size = 0;
    //     const char *c = list;
    //     while (*c != '\0' && size < 8) 
    //     {
    //         if (*c == ',') 
    //         {
    //             c++;
    //             continue;
    //         }
    //         device_ids_.push_back(atoi(c));
    //         size++;
    //         c++;
    //     }
    // }
}

//Starts thread from the ThreadPool
void GPUThreadPool::start()
{
    size_t threads_num = conf->get_int_config("--worker-threads-count=");
    bool cpu_usage_info = conf->get_bool_config("-u");
    std::lock_guard<std::mutex> lk(m_);
    
    if (stop_flag_) {
        throw std::runtime_error("start while stopping ThreadPool");
    }
    
    if (cpu_usage_info) {
        cpu_usage_flags_.resize(std::thread::hardware_concurrency());
    }

    for (auto i = 0u; i < threads_num; i++) {
        threads_.push_back(new std::thread([this, i](){
            this->routine(i);
        }));
    }
}


//Stops thread from the ThreadPool
void GPUThreadPool::stop()
{
    std::unique_lock<std::mutex> lk(m_);
    if (stop_flag_) {
        throw std::runtime_error("stop while stopping ThreadPool");
    }

    stop_flag_ = true;
    cv_.notify_all();

    while (!threads_.empty()) {
        std::thread *t = threads_.back();
        threads_.pop_back();

        lk.unlock();
        t->join();
        delete t;
        lk.lock();
    }

    if (!cpu_usage_flags_.empty()) {
        std::ostringstream cpu_usage_stream;
        cpu_usage_stream << "gpu worker thread pool" << " (cpu usage): [\t";
        for (int i = 0; i < cpu_usage_flags_.size(); ++i) {
            cpu_usage_stream << i << " cpu" << " : " << (cpu_usage_flags_[i] ? "yes" : "no") << "\t";
        }
        cpu_usage_stream << "]\n";
        INFO(cpu_usage_stream.str());
    }

    stop_flag_ = false;
}

void GPUThreadPool::submit(std::function<void()> job, CF *cf_)
{
    std::lock_guard<std::mutex> lk(m_);

    int job_type = 0;
    if (cf_ != nullptr && cf_->get_code_type() == "GPU")
    {
        for (auto param : cf_->get_code_params())
        {
            if (param == "GPU")
            {
                job_type = 1; // 1-GPU
                break;
            }
            if (param == "HYBRID")
            {
                job_type = 2; // 2-HYBRID
                break;
            }
        }
    }

    if (job_type == 1) 
    {
        if (use_gpu_)
        {
            if (num_failed_gpu_threads == device_ids_.size())
            {
                ABORT("Tried to submit GPU job, but none of the threads were able to connect to GPU devices");
            }
            jobs_gpu_.push(job);
        }
        else
        {
            ABORT("Tried to submit GPU job, but --use-gpu was not specified or error occurred during GPU initialization");
        }
    }
    else if (job_type == 2)
    {
        jobs_hybrid_.push(job);
    }
    else
    {
        jobs_.push(job);
    }

    on_submit_handler_();
    cv_.notify_one();
}

thread_local int device_type = 0;
thread_local cudaStream_t thread_cuda_stream = nullptr;
thread_local GPUTimer *thread_timer = nullptr;

void GPUThreadPool::routine(int id)
{
    int device_id = -1;
    bool gpu_mode = false;
    cudaStream_t local_stream = nullptr;

    if (use_gpu_ && id < device_ids_.size()) 
    {
        device_id = device_ids_[id];
        if (cudaSetDevice(device_id) != cudaSuccess) 
        {
            printf("[GPU ERROR] Set device [%d] failed for thread %d. Associated thread will switch to normal mode.\n", 
                   device_id, id);
            
            std::unique_lock<std::mutex> lk(m_);
            num_failed_gpu_threads++;
        }
        else {
            if (cudaStreamCreate(&local_stream) != cudaSuccess)
            {
                printf("[GPU ERROR] Init stream failed for thread %d. Associated thread will switch to normal mode.\n", id);
                cudaDeviceReset();
                local_stream = nullptr;
                
                std::unique_lock<std::mutex> lk(m_);
                num_failed_gpu_threads++;
            }
            else
            {
                std::cout << "[GPU] Set device [" << device_id << "] success for thread " << id << 
                    ". Associated thread will work in GPU mode." << std::endl;
                gpu_mode = true;
            }
        }
    }

    device_type = gpu_mode ? 1 : 0;
    thread_cuda_stream = local_stream;

    // if (use_gpu_timer_)
    // {
    //     thread_timer = new GPUTimer();
    // }

    int gpu_cpu_jobs_skipped = 0;

    std::unique_lock<std::mutex> lk(m_);
    while (!stop_flag_ || !jobs_.empty() || !jobs_hybrid_.empty() ||
            (gpu_mode && !jobs_gpu_.empty()) || running_jobs_ > 0) {
        
        if (jobs_.empty() && jobs_hybrid_.empty() && (!gpu_mode || jobs_gpu_.empty())) {
            on_empty_handler_();
            cv_.wait(lk);
            continue;
        }

        std::function<void()> job;
        
        if (gpu_mode)
        {
            if (!jobs_gpu_.empty())
            {
                gpu_cpu_jobs_skipped = 0;
                job = jobs_gpu_.front();
                jobs_gpu_.pop();
            }
            else if (!jobs_hybrid_.empty())
            {
                gpu_cpu_jobs_skipped = 0;
                job = jobs_hybrid_.front();
                jobs_hybrid_.pop();
            }
            else
            {
                bool skip_job = true;
                if ((threads_.size() == device_ids_.size() && num_failed_gpu_threads == 0) ||
                    gpu_cpu_jobs_skipped > 2)
                {
                    skip_job = false;
                }

                if (!skip_job)
                {
                    gpu_cpu_jobs_skipped = 0;
                    job = jobs_.front();
                    jobs_.pop();
                }
                else 
                {
                    gpu_cpu_jobs_skipped++;
                    cv_.notify_one();
                    cv_.wait(lk);
                    continue;
                }
            }
        }
        else
        {
            if (!jobs_.empty()) {
                job = jobs_.front();
                jobs_.pop();
            }
            else {
                job = jobs_hybrid_.front();
                jobs_hybrid_.pop();
            }
        }

        running_jobs_++;
        lk.unlock();

        job();

        lk.lock();
        running_jobs_--;
        
        if (running_jobs_ == 0 && stop_flag_) {
            cv_.notify_all();
        }
    }

    if (gpu_mode)
    {
        if (cudaStreamDestroy(local_stream) != cudaSuccess)
        {
            printf("[GPU ERROR] Destroy stream for thread [%d] failed.\n", id);
        }
        if (cudaDeviceReset() != cudaSuccess) 
        {
            printf("[GPU ERROR] Reset device [%d] failed.\n", device_id);
        }
    }

    if (thread_timer != nullptr)
    {
        std::lock_guard<std::mutex> guard(timer_mutex);
        if (gpu_mode)
        {
            printf("[GPU Timer] COPY_TO_GPU: %lf.\n", 
                   thread_timer->seconds(GPUTimer::TimePoints::COPY_TO_GPU));
            printf("[GPU Timer] STREAM_SYNCH: %lf.\n", 
                   thread_timer->seconds(GPUTimer::TimePoints::STREAM_SYNCH));
            printf("[GPU Timer] COPY_FROM_GPU: %lf.\n", 
                   thread_timer->seconds(GPUTimer::TimePoints::COPY_FROM_GPU));
            fflush(stdout);
        }
        else
        {
            printf("[CPU Timer] OPERATOR_EXEC: %lf.\n", 
                   thread_timer->seconds(GPUTimer::TimePoints::OPERATOR_EXEC));
            fflush(stdout);
        }

        delete thread_timer;
        thread_timer = nullptr;
    }
}