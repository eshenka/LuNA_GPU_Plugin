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

    // Plugin interface methods
    void set_default_flags(Config *config) override;
    bool accept_flag(std::string flag, Config *config) override;
    bool init(const Config *config) override;
    bool destroy(const Config *config) override;
    const std::string &get_name() override;
    AbstractThreadPool *create_thread_pool(const Config *conf_) override;

    // Singleton instance
    static AbstractThreadPoolPlugin *instance;

private:
    std::string name;
};

// Export C-style factory function
extern "C" {
    AbstractThreadPoolPlugin *gpu_threadpool_create_plugin();
}

#endif // GPU_THREAD_POOL_PLUGIN_H