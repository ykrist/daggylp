#![allow(warnings)]
use quote::ToTokens;

macro_rules! parse_error {
    ($tokens:expr => $msg:expr) => {
      return Err(syn::Error::new_spanned(&$tokens, $msg))
    };
}

macro_rules! impl_parse_for_updatefrommeta  {
    ($t:path) => {
      impl Parse for $t {
        fn parse(input: ParseStream) -> Result<Self> {
          let meta_list = input.call(syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated)?;
          let mut new = <$t>::default();
          for meta in meta_list.iter() {
            if new.update(meta)? {
              continue;
            }
            parse_error!(meta => "unrecognised attribute item")
          }
          Ok(new)
        }
      }
    };
}

mod util;
pub(crate) use util::*;

mod graph_test;
use graph_test::GraphTest;

mod graph_proptest;


#[proc_macro_attribute]
pub fn graph_test(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let test_cases = syn::parse_macro_input!(item as GraphTest);
  test_cases.to_token_stream().into()
  // Default::default()
}

#[proc_macro_attribute]
pub fn graph_proptest(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let mut proptest = syn::parse_macro_input!(item as graph_proptest::GraphProptest);
  let global_settings = syn::parse_macro_input!(attr as graph_proptest::GlobalSettings);
  proptest.global_settings = Some(global_settings);

  proptest.to_token_stream().into()

}