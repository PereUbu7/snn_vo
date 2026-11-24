#include <chrono>
class Time
{
    using nanoseconds = std::chrono::nanoseconds;
    public:
        Time();
        ~Time();
        void Tick()
        {
            lastTime = currentTime;
            currentTime = std::chrono::duration_cast<nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        }

        nanoseconds GetLatestDeltaTime() const
        {
            return currentTime - lastTime;
        }

    private:
        nanoseconds currentTime;
        nanoseconds lastTime;
};