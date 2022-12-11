use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub unsafe extern "C" fn print_ocr(data: *const c_char, label: *const c_char) {
    let data = CStr::from_ptr(data)
        .to_str()
        .expect("failure in ffi interface");
    println!(
        "{}: {}",
        CStr::from_ptr(label).to_string_lossy(),
        advent_of_code_ocr::parse_string_to_letters(data)
    );
}
