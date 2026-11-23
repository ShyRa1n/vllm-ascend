#ifndef CV60N_STATE_INDEX_H
#define CV60N_STATE_INDEX_H

#include <cstdint>

// Pure integer index rules shared by the actual device path and CPU properties.
// Callers validate stateSlots > 0, used >= 0 and nonnegative positions.
namespace cv60n_state {
__aicore__ inline int64_t TailRows(int64_t used, int64_t stateSlots) {
    return used < stateSlots ? used : stateSlots;
}

__aicore__ inline int64_t Slot(int64_t position, int64_t stateSlots) {
    return position % stateSlots;
}

__aicore__ inline int64_t StateRow(int64_t page, int64_t position, int64_t stateSlots) {
    return page * stateSlots + Slot(position, stateSlots);
}

__aicore__ inline int64_t TransferSegment(int64_t used, int64_t stateSlots) {
    const int64_t tail = TailRows(used, stateSlots);
    const int64_t segment = stateSlots - 1 + stateSlots * ((used - tail) / stateSlots);
    return segment >= used - tail && segment < used ? segment : -1;
}

__aicore__ inline int64_t FirstRun(int64_t count, int64_t slot, int64_t stateSlots) {
    return count < stateSlots - slot ? count : stateSlots - slot;
}
}  // namespace cv60n_state

#endif
