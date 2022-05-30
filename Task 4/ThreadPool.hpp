#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <future>

// для методов можно использовать std::mem_fn
class ThreadPool
{
public:
	explicit ThreadPool(unsigned threadsNumber = std::thread::hardware_concurrency())
		: stop(false)
	{
		executors.reserve(threadsNumber);
		
		// добавляем потоки, обрабатывающие задачи, в контейнер 
		for (unsigned i = 0; i < threadsNumber; i++)
		{
			executors.emplace_back(
				[this]
				{
					while (true)
					{
						std::function<void()> task;
						{
							std::unique_lock<std::mutex> lock(queueMutex);
							condition.wait(lock, [this] {return stop || !tasks.empty(); });

							if (stop && tasks.empty())
							{
								return;
							}
							
							// если есть, что выполнять, выполняем
							task = std::move(tasks.front());
							tasks.pop();
						}
						task();
					}
				});
		}
	}

	// добавление задачи в очередь
	template<typename F, class... Args>
	auto Enqueue(F && f, Args && ...args)
	{
		using ResultType = decltype(f(args...));
		
		// создаём задачу
		auto task = std::make_shared<std::packaged_task<ResultType()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
		// создаём future и добавляем задачу в очередь
		std::future<ResultType> result = task->get_future();
		{
			std::unique_lock lock(queueMutex);

			if (stop)
			{
				throw std::runtime_error("Enqueue on stopped ThreadPool");
			}

			tasks.emplace([task]() { (*task)(); });
		}
		// уведомляем, что что-то добавили
		condition.notify_one();
		return result;
	}

	~ThreadPool()
	{
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			stop = true;
		}
		// уведомляет все потоки, что пора заканчивать, т.к. stop == true
		condition.notify_all();
		for (auto & worker : executors)
		{
			worker.join();
		}
	}

private:
	// потоки, которые выполняют задачи
	std::vector<std::thread> executors;
	// сами задачи
	std::queue<std::function<void()>> tasks;
	
	std::mutex queueMutex;
	// условная переменная, который позволяет потокам ждать очередной задачи
	std::condition_variable condition;
	// флаг для остановки
	bool stop;
};
