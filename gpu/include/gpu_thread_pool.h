#pragma once

#include "abstract_thread_pool.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <string>

#include "config.h"

class CF;
class GPUTimer;

class GPUThreadPool : public AbstractThreadPool {
public:
    GPUThreadPool(const Config* config);
    virtual ~GPUThreadPool();

    virtual void start();
    virtual void stop();
    virtual void submit(std::function<void()>, CF*);
    virtual void on_empty(std::function<void()>) { on_empty_handler_ = std::move(on_empty_handler_); }
    virtual void on_submit(std::function<void()>) { on_submit_handler_ = std::move(on_submit_handler_); }
    virtual bool has_jobs();
    virtual std::string to_string() const;

private:
    void routine(int id);
    void set_gpu_related_params(const Config* config);

    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::thread*> threads_;
    std::queue<std::function<void()>> jobs_;
    std::queue<std::function<void()> > jobs_ascend_;
    std::queue<std::function<void()> > jobs_hybrid_;
    std::function<void()> on_empty_handler_;
    std::function<void()> on_submit_handler_;
    size_t running_jobs_;
    bool stop_flag_;
    const Config* conf;

    bool use_gpu_;
    bool use_gpu_timer_;
    std::string gpu_id_list_;
};
