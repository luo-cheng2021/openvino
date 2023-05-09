#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <sched.h>
#include <unistd.h>
#include <memory.h>

class ThreadPool {
public:
    ThreadPool(size_t);
    void enqueue(std::function<void(size_t)> f);

    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::vector<int8_t> start_work;
    std::function<void(size_t)> task;
    size_t threads_num;
    std::atomic<size_t> completed;
    std::atomic<size_t> progress;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false), start_work(threads, false), threads_num(threads)
{
    cpu_set_t my_set;        /* Define your cpu_set bit mask. */
    CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
    CPU_SET(0, &my_set);     /* set the bit that represents core 7. */
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

    for(size_t i = 1;i<threads;++i)
        workers.emplace_back(
            [this, i]
            {
                cpu_set_t my_set;        /* Define your cpu_set bit mask. */
                CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
                CPU_SET(i, &my_set);     /* set the bit that represents core 7. */
                sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
                size_t cur_prg = 0;
                for(;;)
                {
                    {
                        while (this->progress == cur_prg);// asm("pause");

                        if(this->stop)
                            return;
                    }

                    this->task(i);
                    completed += 1;
                    cur_prg++;
                }
            }
        );
}

inline void ThreadPool::enqueue(std::function<void(size_t)> f) 
{
    completed = 0;
    {
        this->task = f;
        progress += 1;
    }
    f(0);
    while(completed.load() != threads_num - 1);

    return;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
        progress += 1;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

#endif