use syn::*;
use syn::parse::{ParseStream, Parse};
use quote::{quote, ToTokens};
use proc_macro2::{Span, TokenStream};

macro_rules! map_lit_to_variant {
    ($name:ident; $($s:literal => $path:path),+ $(,)? ) => {
      pub fn $name(lit: &Lit) -> Result<TokenStream> {
          let value_error_msg = concat!(
            "expected one of: ",
            $($s, " ",)*
          );

          match lit {
            Lit::Str(lit) => {

              let mut s = lit.value();
              s.make_ascii_lowercase();
              match &*s {
                $(
                  $s => Ok(quote! { $path }),
                )*
                _ => Err(Error::new_spanned(lit, value_error_msg)),
              }
            }
            _ => Err(syn::Error::new_spanned(lit, "must be a string literal")),
          }
      }
    };
}

map_lit_to_variant!{ scc_viz_variant;
  "hide" => crate::viz::SccViz::Hide,
  "show" => crate::viz::SccViz::Show,
  "collapse" => crate::viz::SccViz::Collapse,
}

map_lit_to_variant!{ layout_algo_variant;
  "dot" => crate::viz::LayoutAlgo::Dot,
  "fdp" => crate::viz::LayoutAlgo::Fdp,
  "neato" => crate::viz::LayoutAlgo::Neato,
}



#[derive(Copy, Clone)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum TestFnWrapper {
  Simple,
  WithMeta,
  WithData,
  Complex,
}

fn check_mut_graph_arg(arg: &FnArg) -> bool {
  let arg = match arg {
    syn::FnArg::Receiver(_) => return false,
    syn::FnArg::Typed(arg) => arg,
  };
  let arg = match &*arg.ty {
    syn::Type::Reference(arg) => arg,
    _ => return false,
  };
  if arg.mutability.is_none() {
    return false;
  }
  let arg = match &*arg.elem {
    syn::Type::Path(p) => p,
    _ => return false
  };
  arg.path.is_ident("Graph")
}

fn check_graphspec_arg(arg: &FnArg) -> bool {
  let arg = match arg {
    syn::FnArg::Receiver(_) => return false,
    syn::FnArg::Typed(arg) => arg,
  };
  let arg = match &*arg.ty {
    syn::Type::Reference(arg) => arg,
    _ => return false,
  };
  if arg.mutability.is_some() {
    return false;
  }
  let arg = match &*arg.elem {
    syn::Type::Path(p) => p,
    _ => return false
  };
  arg.path.is_ident("GraphData")
}

fn check_meta_arg(arg: &FnArg) -> bool {
  match arg {
    syn::FnArg::Receiver(_) => false,
    syn::FnArg::Typed(_) => true,
  }
}

fn check_output_ty(ret: &ReturnType, expected: &str) -> Result<()> {
  let error = || Err(syn::Error::new_spanned(ret, format!("expected return type: {}", expected)));

  let ret = match &*ret {
    ReturnType::Type(_, ty) => ty,
    _ => return error(),
  };

  let ret = match &**ret {
    Type::Path(p) => p,
    _ => return error()
  };

  if !ret.path.is_ident(expected) {
    return error();
  };

  Ok(())
}

impl TestFnWrapper {
  pub fn from_signature(sig: &Signature, output_ty: &str) -> Result<Self> {
    if sig.abi.is_some() {
      parse_error!(sig.abi => "Cannot specify ABI in test fn");
    }
    if sig.unsafety.is_some() {
      parse_error!(sig.unsafety => "Test fn cannot be unsafe");
    }
    if sig.asyncness.is_some() {
      parse_error!(sig.asyncness => "Test fn cannot be async")
    }

    check_output_ty(&sig.output, output_ty)?;

    let args_error = || {
      Err(syn::Error::new_spanned(
        &sig.inputs,
        "Invalid test fn arguments: expected fn(&mut Graph), fn(&mut Graph, &GraphSpec), fn(&mut Graph, _) or fn(&mut Graph, &GraphSpec, _)",
      ))
    };

    let inputs: Vec<_> = sig.inputs.iter().collect();
    if !(1..=3).contains(&inputs.len()) {
      return args_error();
    }

    if !check_mut_graph_arg(&inputs[0]) {
      return args_error();
    }

    if inputs.len() == 1 {
      return Ok(TestFnWrapper::Simple);
    } else if check_graphspec_arg(&inputs[1]) {
      if inputs.len() == 2 {
        return Ok(TestFnWrapper::WithData);
      } else if check_meta_arg(&inputs[2]) {
        return Ok(TestFnWrapper::Complex);
      }
    } else if inputs.len() == 2 && check_meta_arg(&inputs[1]) {
      return Ok(TestFnWrapper::WithMeta);
    }

    args_error()
  }
}


impl ToTokens for TestFnWrapper {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    use TestFnWrapper::*;
    match self {
      Simple =>
        tokens.extend(quote! { crate::test::SimpleTest }),
      WithMeta =>
        tokens.extend(quote! { crate::test::TestWithMeta }),
      WithData =>
        tokens.extend(quote! { crate::test::TestWithData }),
      Complex =>
        tokens.extend(quote! { crate::test::ComplexTest }),
    }
  }
}

pub trait Directive: Sized {
  fn from_attribute(attr: &Attribute) -> Result<Option<Self>>;
}


pub trait UpdateFromMeta {
  fn update(&mut self, meta: &Meta) -> Result<bool>;
}

#[derive(Default, Clone)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct VizConfig {
  sccs: Option<TokenStream>,
  layout: Option<TokenStream>,
}

impl ToTokens for VizConfig {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.extend(quote! {
      let viz_config = runner.viz_config();
    });
    if let Some(v) = &self.sccs {
      tokens.extend(quote! {
        viz_config.sccs(#v);
      })
    }
    if let Some(v) = &self.layout {
      tokens.extend(quote! {
        viz_config.layout(#v);
      })
    }
  }
}

impl UpdateFromMeta for VizConfig {
  fn update(&mut self, meta: &Meta) -> Result<bool> {
    match meta {
      Meta::NameValue(kwarg) => {
        if kwarg.path.is_ident("sccs") {
          self.sccs = Some(scc_viz_variant(&kwarg.lit)?);
        } else if kwarg.path.is_ident("layout") {
          self.layout = Some(layout_algo_variant(&kwarg.lit)?);
        } else {
          return Ok(false);
        }
        Ok(true)
      }
      _ => Ok(false)
    }
  }
}

impl_parse_for_updatefrommeta!(VizConfig);

#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Test<D, S> {
  pub global_settings: Option<S>,
  pub outer_attrs: Vec<syn::Attribute>,
  pub directives: Vec<D>,
  pub fn_ident: syn::Ident,
  pub fn_wrapper: TestFnWrapper,
  pub fn_sig: syn::Signature,
  pub fn_body: syn::ExprBlock,
}

impl<D: Directive, S> Test<D, S> {
  pub fn parse(input: ParseStream, expected_output: &str) -> Result<Self> {
    let mut directives = Vec::default();
    let mut outer_attrs: Vec<syn::Attribute> = input.call(syn::Attribute::parse_outer)?;
    let mut err = Ok(());

    outer_attrs.retain(|attr| {
      if err.is_err() { return true }
      match D::from_attribute(attr) {
        Err(e) => {
          err = Err(e);
          true
        },
        Ok(Some(d)) => {
          directives.push(d);
          false
        },
        Ok(None) => true,
      }
    });

    err?;

    let mut fn_sig: syn::Signature = input.parse()?;
    let fn_body: syn::ExprBlock = input.parse()?;

    let fn_wrapper = TestFnWrapper::from_signature(&fn_sig, expected_output)?;
    let fn_ident = std::mem::replace(
      &mut fn_sig.ident,
      syn::Ident::new("test_fn", Span::call_site()),
    );

    Ok(Test {
      global_settings: None,
      outer_attrs,
      directives,
      fn_ident,
      fn_wrapper,
      fn_sig,
      fn_body,
    })
  }
}

