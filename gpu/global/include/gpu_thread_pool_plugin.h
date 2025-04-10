#ifndef GPU_THREAD_POOL_PLUGIN_H
#define GPU_THREAD_POOL_PLUGIN_H

#include <string>
#include "abstract_thread_pool_plugin.h"

class Config;
class AbstractThreadPool;

class GPUThreadPoolPlugin : public AbstractThreadPoolPlugin {
public:
    GPUThreadPoolPlugin();
    ~GPUThreadPoolPlugin();

    virtual void set_default_flags(Config *config);
    virtual bool accept_flag(std::string flag, Config *config);
    virtual bool init(const Config *config);
    virtual bool destroy(const Config *config);
    virtual const std::string &get_name();
    virtual AbstractThreadPool *create_thread_pool(const Config *conf_);
    

private:
    static AbstractThreadPoolPlugin *instance;
    std::string name = "gpu_threadpool";
};

#endif // GPU_THREAD_POOL_PLUGIN_H