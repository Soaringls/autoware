#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace noncopyable_{
class noncopyable{
 protected:
  noncopyable(){}
  ~noncopyable(){}
 private:
  noncopyable(const noncopyable& );
  const noncopyable& operator=(const noncopyable&);
};
}

template<typename T>
class SingleInstance{
 public:
  template<typename... Args>
  static T& GetInstance(Args &&... args){
      if(instance_ == nullptr){
          instance_.reset(new T(std::forward<Args>(args)...));
      }
    //   std::call_once(
    //       get_once_flag(),
    //       [](Args &&... args){
    //           instance_.reset(new T(std::forward<Args>(args)...));
    //       },
    //       std::forward<Args>(args)...
    //   );
      return *instance_.get();
  }
 private:
  static std::unique_ptr<T> instance_;
  static std::once_flag& get_once_flag(){
      static std::once_flag once_;
      return once_;
  }
};

template <class T>
std::unique_ptr<T> SingleInstance<T>::instance_ = nullptr;
