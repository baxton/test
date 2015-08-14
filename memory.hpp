

#if !defined MEMORY_DOT_HPP
#define MEMORY_DOT_HPP



namespace ma {

namespace memory {


template<class T>
class ptr_vec {
    T* p_;

public:
    ptr_vec() : p_(NULL) {}
    ptr_vec(ptr_vec& other) : p_(NULL) {
        swap(other);
    }
    ptr_vec(T* p) : p_(p) {}

    ~ptr_vec() {
        free();
    }

    ptr_vec& operator= (ptr_vec& other) {
        swap(other);
        return *this;
    }

    void reset(T* p) {
        free();
        p_ = p;
    }

    void swap(ptr_vec& other) {
        T* tmp = other.p_;
        other.p_ = p_;
        p_ = tmp;
    }

    T& operator[] (int i) {
        return p_[i];
    }

    void free() {
        if (p_)
            delete [] p_;
        p_ = NULL;
    }

    T* get() const {return p_;}
};


}
}
#endif
